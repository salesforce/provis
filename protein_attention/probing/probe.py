"""Train and evaluate probing classifiers. Based on training.py from TAPE, modified to perform probing specifically."""

import inspect
import json
import logging
import os
import typing
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from tape.datasets import ProteinnetDataset
from tape import errors
from tape import utils
from tape import visualization
from tape.metrics import accuracy
from tape.models.modeling_utils import ProteinModel
from tape.optimization import WarmupLinearSchedule
from torch.utils.data import DataLoader
from tqdm import tqdm

from protein_attention.datasets import SecondaryStructureOneVsAllDataset, BindingSiteDataset
from protein_attention.probing.metrics import precision, recall, f1, precision_at_ks
from protein_attention.probing.models import ProteinBertForLinearSequenceToSequenceProbing, ProteinBertForContactProbing
from protein_attention.utils import get_data_path

try:
    from apex import amp
    import amp_C
    import apex_C
    from apex.amp import _amp_state
    from apex.parallel.distributed import flat_dist_call
    from apex.parallel.distributed import DistributedDataParallel as DDP

    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]
OutputDict = typing.Dict[str, typing.Any]


class ForwardRunner:

    def __init__(self,
                 model: ProteinModel,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.local_rank = local_rank

        forward_arg_keys = inspect.getfullargspec(model.forward).args
        forward_arg_keys = forward_arg_keys[1:]  # remove self argument
        self._forward_arg_keys = forward_arg_keys
        assert 'input_ids' in self._forward_arg_keys

    def initialize_distributed_model(self):
        if self.local_rank != -1:
            if not self.fp16:
                self.model = DDP(self.model)
            else:
                flat_dist_call([param.data for param in self.model.parameters()],
                               torch.distributed.broadcast, (0,))
        elif self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False,
                no_loss: bool = False):
        # Filter out batch items that aren't used in this model
        # Requires that dataset keys match the forward args of the model
        # Useful if some elements of the data are only used by certain models
        # e.g. PSSMs / MSAs and other evolutionary data
        batch = {name: tensor for name, tensor in batch.items()
                 if name in self._forward_arg_keys}
        if self.device.type == 'cuda':
            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}

        outputs = self.model(**batch)

        if no_loss:
            return outputs

        if isinstance(outputs[0], tuple):
            # model also returned metrics
            loss, metrics = outputs[0]
        else:
            # no metrics
            loss = outputs[0]
            metrics = {}

        if self.n_gpu > 1:  # pytorch DataDistributed doesn't mean scalars
            loss = loss.mean()
            metrics = {name: metric.mean() for name, metric in metrics.items()}

        if return_outputs:
            return loss, metrics, outputs
        else:
            return loss, metrics

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self


class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: ProteinModel,
                 optimizer: optim.Optimizer,  # type: ignore
                 gradient_accumulation_steps: int = 1,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1,
                 max_grad_norm: float = 1.0,
                 warmup_steps: int = 0,
                 num_train_optimization_steps: int = 1000000):

        super().__init__(model, device, n_gpu, fp16, local_rank)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self._global_step = 0
        self._local_rank = local_rank
        self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._delay_accumulation = fp16 and local_rank != -1

        self.scheduler = WarmupLinearSchedule(
            self.optimizer, warmup_steps, num_train_optimization_steps)

    def initialize_fp16(self):
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O2", loss_scale="dynamic",
                master_weights=True)
            _amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    def resume_from_checkpoint(self, checkpoint_dir: str) -> int:
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint.bin'), map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.fp16:
            self.optimizer._lazy_init_maybe_master_weights()
            self.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved in zip(
                    amp.master_params(self.optimizer), checkpoint['master params']):
                param.data.copy_(saved.data)
            amp.load_state_dict(checkpoint['amp'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch

    def save_state(self, save_directory: typing.Union[str, Path], epoch_id: int):
        save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir()
        else:
            assert save_directory.is_dir(), "Save path should be a directory"
        model_to_save = getattr(self.model, 'module', self.model)
        model_to_save.save_pretrained(save_directory)
        optimizer_state: typing.Dict[str, typing.Any] = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch_id}
        if APEX_FOUND:
            optimizer_state['master params'] = list(amp.master_params(self.optimizer))
            try:
                optimizer_state['amp'] = amp.state_dict()
            except AttributeError:
                pass
        torch.save(optimizer_state, save_directory / 'checkpoint.bin')

    def backward(self, loss) -> None:
        if not self._delay_accumulation:
            loss = loss / self.gradient_accumulation_steps
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer,
                                delay_overflow_check=self._delay_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self._local_rank == -1:
            self._step()
        elif not self.fp16:
            self._step()
        else:
            self._step_distributed_fp16()

    def _step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore
        self._global_step += 1

    def _step_distributed_fp16(self) -> None:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        # allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else \
        # torch.float32
        allreduce_dtype = torch.float16
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (
                    torch.distributed.get_world_size() * self.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [allreduced_views, master_grads],
            1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = self._overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            self._step()
        else:
            # Overflow detected, print message and clear gradients
            logger.info(f"Gradient overflow.  Skipping step, reducing loss scale to "
                        f"{scaler.loss_scale()}")
            if _amp_state.opt_properties.master_weights:
                for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in self.model.parameters():
            param.grad = None

    @property
    def global_step(self) -> int:
        return self._global_step


def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    runner: BackwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    num_log_iter: int = 20,
                    gradient_accumulation_steps: int = 1) -> LossAndMetrics:
    if viz is None:
        viz = visualization.DummyVisualizer()
    smoothing = 1 - 1 / num_log_iter
    accumulator = utils.MetricsAccumulator(smoothing)

    torch.set_grad_enabled(True)
    runner.train()

    def make_log_str(step: int, time: float) -> str:
        ep_percent = epoch_id + step / len(train_loader)
        if runner.scheduler is not None:
            curr_lr = runner.scheduler.get_lr()[0]  # type: ignore
        else:
            curr_lr = runner.optimizer.param_groups[0]['lr']

        print_str = []
        print_str.append(f"[Ep: {ep_percent:.2f}]")
        print_str.append(f"[Iter: {runner.global_step}]")
        print_str.append(f"[Time: {time:5.2f}s]")
        print_str.append(f"[Loss: {accumulator.loss():.5g}]")

        for name, value in accumulator.metrics().items():
            print_str.append(f"[{name.capitalize()}: {value:.5g}]")

        print_str.append(f"[LR: {curr_lr:.5g}]")
        return ''.join(print_str)

    start_t = timer()
    for step, batch in enumerate(train_loader):
        loss, metrics = runner.forward(batch)  # type: ignore
        runner.backward(loss)
        accumulator.update(loss, metrics, step=False)
        if (step + 1) % gradient_accumulation_steps == 0:
            runner.step()
            viz.log_metrics(accumulator.step(), "train", runner.global_step)
            if runner.global_step % num_log_iter == 0:
                end_t = timer()
                logger.info(make_log_str(step, end_t - start_t))
                start_t = end_t

    final_print_str = f"Train: [Loss: {accumulator.final_loss():.5g}]"
    for name, value in accumulator.final_metrics().items():
        final_print_str += f"[{name.capitalize()}: {value:.5g}]"
    logger.info(final_print_str)
    return accumulator.final_loss(), accumulator.final_metrics()


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    is_master: bool = True) -> typing.Tuple[float, typing.Dict[str, float]]:
    num_batches = len(valid_loader)
    accumulator = utils.MetricsAccumulator()

    torch.set_grad_enabled(False)
    runner.eval()

    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        accumulator.update(loss, metrics)

    # Reduce loss across all processes if multiprocessing
    eval_loss = utils.reduce_scalar(accumulator.final_loss())
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"
    for name, value in metrics.items():
        print_str += f"[{name.capitalize()}: {value:.5g}]"

    metrics['loss'] = eval_loss
    if viz is not None:
        viz.log_metrics(metrics, "val", getattr(runner, 'global_step', epoch_id))

    logger.info(print_str)

    return eval_loss, metrics


def _get_outputs_to_save(batch, outputs):
    targets = batch['targets'].cpu().numpy()
    outputs = outputs.cpu().numpy()
    protein_length = batch['protein_length'].sum(1).cpu().numpy()

    reshaped_output = []
    for target, output, plength in zip(targets, outputs, protein_length):
        output_slices = tuple(slice(1, plength - 1) if dim == protein_length.max() else
                              slice(0, dim) for dim in output.shape)
        output = output[output_slices]
        target = target[output_slices]

        reshaped_output.append((target, output))
    reshaped_output


def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   get_sequence_lengths: bool,
                   is_master: bool = True, ) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()

    save_outputs = []
    sequence_lengths = []

    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        predictions = outputs[1].cpu().numpy()
        targets = batch['targets'].cpu().numpy()
        for pred, target in zip(predictions, targets):
            save_outputs.append({'prediction': pred, 'target': target})
        if get_sequence_lengths:
            sequence_lengths.extend(batch['protein_length'].tolist())

    if get_sequence_lengths:
        assert len(sequence_lengths) == len(save_outputs)
        return save_outputs, sequence_lengths
    else:
        return save_outputs


def run_train(task: str,
              num_hidden_layers: int,
              one_vs_all_label: str = None,
              label_scheme: str = None,
              learning_rate: float = 1e-4,
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              num_workers: int = 0,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1) -> None:
    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    data_dir = get_data_path()
    output_dir = data_dir / 'probing'
    exp_dir = f'{(exp_name + "_") if exp_name else ""}{task}_{(one_vs_all_label + "_") if one_vs_all_label else ""}' \
              f'{num_hidden_layers}'
    save_path = Path(output_dir) / exp_dir

    if is_master:
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

    if task == 'secondary':
        num_labels = 2
        model = ProteinBertForLinearSequenceToSequenceProbing.from_pretrained('bert-base',
                                                                              num_hidden_layers=num_hidden_layers,
                                                                              num_labels=num_labels)
        if label_scheme == 'ss4':
            label = int(one_vs_all_label)
        else:
            label = one_vs_all_label
        train_dataset = SecondaryStructureOneVsAllDataset(data_dir, 'train', label_scheme, label)
        valid_dataset = SecondaryStructureOneVsAllDataset(data_dir, 'valid', label_scheme, label)
    elif task == 'binding_sites':
        num_labels = 2
        model = ProteinBertForLinearSequenceToSequenceProbing.from_pretrained('bert-base',
                                                                              num_hidden_layers=num_hidden_layers,
                                                                              num_labels=num_labels)
        train_dataset = BindingSiteDataset(data_dir, 'train')
        valid_dataset = BindingSiteDataset(data_dir, 'valid')
    elif task == 'contact_map':
        num_labels = 2
        model = ProteinBertForContactProbing.from_pretrained('bert-base', num_hidden_layers=num_hidden_layers)
        train_dataset = ProteinnetDataset(data_dir, 'train')
        valid_dataset = ProteinnetDataset(data_dir, 'valid')
    else:
        raise NotImplementedError

    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    train_loader = utils.setup_loader(
        train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()

    start_epoch = 0
    runner.initialize_distributed_model()

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs = %d", num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    best_val_loss = float('inf')
    num_evals_no_improvement = 0

    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        if not is_master:
            return False
        if isinstance(save_freq, int):
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)
        else:
            return num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_functions = [accuracy, precision, recall, f1]

    # ACTUAL TRAIN/EVAL LOOP #
    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(start_epoch, num_train_epochs):
            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps)
            if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                val_loss, metric = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    num_evals_no_improvement = 0
                    if task == 'contact_map':
                        outputs, seq_lens = run_eval_epoch(valid_loader, runner, get_sequence_lengths=True)
                    else:
                        outputs = run_eval_epoch(valid_loader, runner, get_sequence_lengths=False)
                    target = [el['target'] for el in outputs]
                    prediction = [el['prediction'] for el in outputs]

                    if task == 'contact_map':
                        # Reshape 2d to 1d
                        prediction = [torch.tensor(prediction_matrix).view(-1, 2).tolist() for prediction_matrix in
                                      prediction]
                        target = [torch.tensor(target_matrix).view(-1).tolist() for target_matrix in target]

                    metrics_to_save = {name: metric(target, prediction)
                                       for name, metric in zip(metrics, metric_functions)}
                    if task == 'contact_map':
                        print(seq_lens)
                        print('first one')
                        print(seq_lens[0])
                        ks = [int(round(seq_len / 5)) for seq_len in seq_lens]
                        metrics_to_save['precision_at_k'] = precision_at_ks(ks, target, prediction)
                    elif task == 'binding_sites':
                        seq_lens = []
                        for target_array in target:
                            mask = target_array != -1
                            seq_lens.append(mask.sum())
                        ks = [int(round(seq_len / 20)) for seq_len in seq_lens]
                        metrics_to_save['precision_at_k'] = precision_at_ks(ks, target, prediction)
                    print(metrics_to_save)
                    metrics_to_save['loss'] = val_loss
                else:
                    num_evals_no_improvement += 1

            # Save trained model
            if do_save(epoch_id, num_evals_no_improvement):
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id)
                logger.info(f"Saving model checkpoint to {save_path}")

            utils.barrier_if_distributed()
            if patience > 0 and num_evals_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_evals_no_improvement} epochs.")
                logger.log(35, f"Best Val Loss: {best_val_loss}")
                if local_rank != -1:
                    # If you're distributed, raise this error. It sends a signal to
                    # the master process which lets it kill other processes and terminate
                    # without actually reporting an error. See utils/distributed_utils.py
                    # for the signal handling code.
                    raise errors.EarlyStopping
                else:
                    break
    logger.info(f"Finished training after {num_train_epochs} epochs.")
    if best_val_loss != float('inf'):
        logger.log(35, f"Best Val Loss: {best_val_loss}")

    with open(save_path / 'results.json', 'w') as outfile:
        json.dump(metrics_to_save, outfile)


if __name__ == "__main__":
    import argparse

    base_parser = argparse.ArgumentParser(description='Parent parser for tape functions',
                                          add_help=False)
    base_parser.add_argument('--no_cuda', action='store_true', help='CPU-only flag')
    base_parser.add_argument('--seed', default=42, type=int, help='Random seed to use')
    base_parser.add_argument('--local_rank', type=int, default=-1,
                             help='Local rank of process in distributed training. '
                                  'Set by launch script.')
    base_parser.add_argument('--num_workers', default=8, type=int,
                             help='Number of workers to use for multi-threaded data loading')
    base_parser.add_argument('--log_level', default=logging.INFO,
                             choices=['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR',
                                      logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
                             help="log level for the experiment")
    base_parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    parser = argparse.ArgumentParser(description='Run Probing on the TAPE datasets',
                                     parents=[base_parser])
    parser.add_argument('task')
    parser.add_argument('--num_hidden_layers', type=int, default=None)
    parser.add_argument('--one_vs_all_label', default=None)
    parser.add_argument('--label_scheme', default=None)
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--num_train_epochs', default=10, type=int,
                        help='Number of training epochs')
    parser.add_argument('--num_log_iter', default=20, type=int,
                        help='Number of training steps per log iteration')
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16 weights')
    parser.add_argument('--warmup_steps', default=10000, type=int,
                        help='Number of learning rate warmup steps')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of forward passes to make for each backwards pass')
    parser.add_argument('--loss_scale', default=0, type=int,
                        help='Loss scaling. Only used during fp16 training.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Maximum gradient norm')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Name to give to this experiment')
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--eval_freq', type=int, default=1,
                        help="Frequency of eval pass. A value <= 0 means the eval pass is "
                             "not run")
    parser.add_argument('--save_freq', default=1, type=utils.int_or_str,
                        help="How often to save the model during training. Either an integer "
                             "frequency or the string 'improvement'")
    parser.add_argument('--patience', default=-1, type=int,
                        help="How many epochs without improvement to wait before ending "
                             "training")

    args = parser.parse_args()

    if args.num_hidden_layers:
        num_hidden_layers_list = [args.num_hidden_layers]
    else:
        num_hidden_layers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for num_hidden_layers in num_hidden_layers_list:
        run_train(task=args.task,
                  num_hidden_layers=num_hidden_layers,
                  label_scheme=args.label_scheme,
                  one_vs_all_label=args.one_vs_all_label,
                  learning_rate=args.learning_rate,
                  batch_size=args.batch_size,
                  num_train_epochs=args.num_train_epochs,
                  num_log_iter=args.num_log_iter,
                  fp16=args.fp16,
                  warmup_steps=args.warmup_steps,
                  gradient_accumulation_steps=args.gradient_accumulation_steps,
                  loss_scale=args.loss_scale,
                  max_grad_norm=args.max_grad_norm,
                  exp_name=args.exp_name,
                  log_dir=args.log_dir,
                  eval_freq=args.eval_freq,
                  save_freq=args.save_freq,
                  no_cuda=args.no_cuda,
                  seed=args.seed,
                  local_rank=args.local_rank,
                  num_workers=args.num_workers,
                  debug=args.debug,
                  log_level=logging.INFO,
                  patience=args.patience)
