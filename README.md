# NILM-Knowledge-Distillation

Il codice implementa una rete RNN basata su GRU per risolvere il task di Non Intrusivce Load Monitoring come rete neurale Teacher. Viene implementata una rete neurale Student addestrata attraverso Knowledge Distillation. 

loss = alpha*Lsoft +(1-alpha)*Lhard


# Use code 
il codice è composto da tre parti principali 
- dataset_manager (preparazione del dataset)
- Train_teacher (Train della teacher_network)
- Train_student (Train della student_network)

Ps. i file .sh automatizzano le operazioni di train per tutte i carichi (fridge washingmachine microwave dishwasher)

# Results
i risultati sono stati ottenuti con:
window = 200
batchsize =100
alpha = 0.8

![](images/washingmachineteacher.png)
![](images/washingmachinestudent.png)
