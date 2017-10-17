#for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
#do
#    for I in `seq 0 29`
#    do
#	python mnist_add_rbf.py mlp_$I mlprbf_${B}_$I --betas $B > mlprbf_${B}_$I.txt 
#    done
#done

for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
do
    for I in `seq 0 15`
    do
	python mnist_add_rbf.py cnn_$I cnnrbf_${B}_$I --betas $B --cnn > cnnrbf_${B}_$I.txt
    done
done
