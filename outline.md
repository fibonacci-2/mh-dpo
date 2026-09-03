# rq1
to what extent english dpo transfer to arabic cbt dialoges?
for this we need to compare performance on arabic cbt task of a model 
that has gone under dpo and its base variant.

Model: https://huggingface.co/Psychotherapy-LLM/PsyCoPref-Llama3-8B


## data
arabic data: data/shifaa-train.csv
use questions only from here. and ask them to the dpo english model and the
base variant. 

# metrics
use same metrics as these papers. the easiest ones to implement to judge quality of cbt response generated 
by model (on the arabic task)
https://huggingface.co/Psychotherapy-LLM/PsyCoPref-Llama3-8B

https://huggingface.co/Psychotherapy-LLM/PsyCoPref-Llama3-8B