#!/bin/bash

#python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_1_rm3.res > ./results/run_1_rm3_eval.txt
#python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_2_bm25.res > ./results/run_2_bm25_eval.txt
#python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_3_qld.res > ./results/run_3_qld_eval.txt
#python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_4_lightgbm.res > ./results/run_4_lightgbm_eval.txt
#python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_5_vector.res > ./results/run_5_vector_eval.txt
python -m pyserini.eval.trec_eval -q ./files/qrels_50_Queries ./results/run_4_lightgbm_pilot.res > ./results/run_4_lightgbm_eval_pilot.txt
