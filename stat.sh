for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
do
    cat mlprbf_${B}_adversarial.txt | python log2stats.py
    echo 
done
