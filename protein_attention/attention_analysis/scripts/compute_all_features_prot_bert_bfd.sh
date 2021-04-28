python compute_edge_features.py \
 --exp-name edge_features_contact_prot_bert_bfd \
 --model-version prot_bert_bfd \
 --features contact_map \
 --dataset proteinnet \
 --num-sequences 5000 \
 --max-seq-len 512 \
 --min-attn .3 \
 --shuffle &&
python compute_edge_features.py  \
 --exp-name edge_features_modifications_prot_bert_bfd \
 --model-version prot_bert_bfd \
 --features protein_modifications \
 --dataset protein_modifications \
 --num-sequences 5000 \
 --max-seq-len 512 \
 --min-attn .3 \
 --shuffle &&
python compute_edge_features.py  \
 --exp-name edge_features_sec_prot_bert_bfd \
 --model-version prot_bert_bfd \
 --features ss4 \
 --dataset secondary \
 --num-sequences 5000 \
 --max-seq-len 512 \
 --min-attn .3 \
 --shuffle &&
python compute_edge_features.py  \
 --exp-name edge_features_sites_prot_bert_bfd \
 --model-version prot_bert_bfd \
 --features binding_sites \
 --dataset binding_sites \
 --num-sequences 5000 \
 --max-seq-len 512 \
 --min-attn .3 \
 --shuffle &&
 python compute_edge_features.py  \
 --exp-name edge_features_aa_prot_bert_bfd \
 --model-version prot_bert_bfd \
 --features aa \
 --dataset proteinnet \
 --num-sequences 5000 \
 --max-seq-len 512 \
 --min-attn .3 \
 --shuffle