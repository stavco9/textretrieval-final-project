#!/bin/bash

python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_1_rm3.res > ./results/run_1_rm3_eval.txt
#python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_2_bm25.res > ./results/run_2_bm25_eval.txt
#python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_3_qld.res > ./results/run_3_qld_eval.txt
