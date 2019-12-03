#!/bin/bash

#sirlm dir location and output dir location
sirlm_loc="/Users/xuefeng/Desktop/hiwi/baayen/srilm/lm/bin/macosx/"
data_loc="./data/"
output_loc="./output/"
corpus="corpus.txt"
test_corpus="test_corpus.txt"
test_corpus2="test_corpus2.txt"
count_file=$corpus".count"
lm_file=$corpus".lm"

#options
order="2"

#count
echo "count..."
count=$sirlm_loc"ngram-count -text "$data_loc$corpus" -order "$order" -write "$output_loc$count_file
$count

#lm
echo "lm..."
lm=$sirlm_loc"ngram-count -text "$data_loc$corpus" -order "$order" -addsmooth 0 -lm "$output_loc$lm_file
$lm

#perplexity
echo "perplexity..."
echo "test 1"
perplexity=$sirlm_loc"ngram -lm "$output_loc$lm_file" -ppl "$data_loc$test_corpus
perplexity=$perplexity" -debug 2"
$perplexity
echo "test 2"
perplexity=$sirlm_loc"ngram -lm "$output_loc$lm_file" -ppl "$data_loc$test_corpus2
perplexity=$perplexity" -debug 2"
$perplexity

#lm
echo "lm addsmooth..."
lm_addsmooth=$sirlm_loc"ngram-count -text "$data_loc$corpus" -order "$order" -addsmooth 1 -lm "$output_loc$lm_file
$lm_addsmooth

#perplexity
echo "perplexity..."
echo "test 1"
perplexity=$sirlm_loc"ngram -lm "$output_loc$lm_file" -ppl "$data_loc$test_corpus
perplexity=$perplexity" -debug 2"
$perplexity
echo "test 2"
perplexity=$sirlm_loc"ngram -lm "$output_loc$lm_file" -ppl "$data_loc$test_corpus2
perplexity=$perplexity" -debug 2"
$perplexity