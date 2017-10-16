#for I in `seq 0 29`
#do
#    python eval_adversarial.py mlp_$I >> mlp_adversarial.txt 
#done

for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
do
    for I in `seq 0 29`
    do
	python eval_adversarial.py mlp_${B}_$I >> mlp_${B}_adversarial.txt 
    done
done
