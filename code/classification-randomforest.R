
# Adresse du dossier où vous travaillez
setwd("C:/Users/as/Desktop/TP_R/Projet/Code")
# Packages utilisés dans la suite
library(class)
# Packages utilisés dans la suite
library(randomForest)
library(caret)
library(ROCR)

# Supprimer toutes les variables
rm(list=ls(all=TRUE))
# Supprimer tous les graphiques déjà présents
graphics.off()
# Lecture des données d’apprentissage
load("../Data/Projets/farm_data_train.rda");
load("../Data/Projets/farm_data_test.rda");



#data_train_x <- data.frame(R2=data_train$R2,R14=data_train$R14,R17=data_train$R17,R32=data_train$R32)

#######kfold

somme<-0.

# Nombre de plis (k)
k <- 15


# Fonction pour effectuer la validation croisée k-fold
validation_kfold <- function(data_train, k) {
  set.seed(123)  # Pour la reproductibilité
  indices <- sample(1:k, nrow(data_train), replace = TRUE)
  
  error_rate <- numeric(k)
  f_score <- numeric(k)
  AUC <-numeric(k)
  
  
  for (i in 1:k) {
    ensemble_test <- data_train[indices == i, ]
    ensemble_apprentissage <- data_train[indices != i, ]
    
    # Séparation des données et de la sortie
    ensemble_apprentissage_x <- data.frame(R2=ensemble_apprentissage$R2,R14=ensemble_apprentissage$R14,R17=ensemble_apprentissage$R17,R32=ensemble_apprentissage$R32)
    ensemble_test_x <- data.frame(R2=ensemble_test$R2,R14=ensemble_test$R14,R17=ensemble_test$R17,R32=ensemble_test$R32)
    #ensemble_apprentissage_y <- ensemble_apprentissage$DIFF
    ensemble_apprentissage_y <- as.factor(ensemble_apprentissage$DIFF)
   
    ####### Modèle d'apprentissage (RF)
    
    
    # Forêts aléatoires
    rf <- randomForest(x = ensemble_apprentissage_x, y = ensemble_apprentissage_y)#,ntre=500,sampsize = 100,maxnodes = 100)
    
    # Prédiction sur l'ensemble de test
    #data_test_predict <- predict(rf, newdata=ensemble_test_x, type="class")
    data_test_predict <- predict(rf, newdata=ensemble_test_x, type="class")
    data_test_predict_with_proba <- predict(rf, newdata=ensemble_test_x, type="prob")
    seuil <-0.6
    data_test_predict <- ifelse( data_test_predict_with_proba[,2] > seuil,"1","0")
    print(data_test_predict)
     
    # Calcul de l'erreur (pour test)
    
    
    error_rate[i] <- mean(data_test_predict != ensemble_test$DIFF)
    
    cat("error_rate using test data = ",error_rate[i],"\n")
    
    # Matrice de confusion
    confmat = table(data_test_predict,ensemble_test$DIFF)
    print("Confusion Matrix")
    print(confmat)
    # vrais positifs + vrais negatifs + faux positifs + faux négatifs
    TP = confmat[1,1]; TN = confmat[2,2]; FP = confmat[1,2]; FN = confmat[2,1];
    
    # Sensibilité (sensitivity ; TPR = true positive rate)
    TPR = TP/(TP+FN)
    cat("TPR",TPR,"\n")
    
    # Spécificité (specificity ; TNR = true negative rate)
    TNR = TN/(TN+FP)
    cat("TNR",TNR,"\n")
    # Précision (precision ; positive predictive value)
    PPV = TP/(TP+FP)
    cat("PPV",PPV,"\n")
    
    # se compare à la prévalence (prevalence)
    cat("Prev =",length(ensemble_test$DIFF[ensemble_test$DIFF==1])/length(ensemble_test$DIFF),"\n")
    
    #cat("F-score = ",2 * TPR * PPV / (TPR+PPV),"\n")
    
    f_score[i]=2 * TPR * PPV / (TPR+PPV)
    
    # k plus proches voisins avec les probas
    
    data_test_predict_with_proba <- predict(rf, newdata=ensemble_test_x, type="class")
    data_test_predict_with_proba_proba <- predict(rf, newdata=ensemble_test_x, type="prob")
  
    
    
    # Calcul du score
    score <- data_test_predict_with_proba_proba[, 2]
    print(score)
    #score <- ifelse(data_test_predict_with_proba == "0", 1-score, score)
    
    pred_rf <- prediction(score, ensemble_test$DIFF)
    perf <- performance(pred_rf, "tpr", "fpr")
    plot(perf,colorize=TRUE)
    par(new=T)
    plot(c(0,1),c(0,1),type="l",ann=FALSE)
    
    # Aire sous la courbe
    AUC[i] <- performance(pred_rf, "auc")@y.values[[1]]
    cat("AUC = ", AUC[i],"\n")
    
    # Choix du seuil
    result <- NULL
    threshold <- seq(0,1,len=11)
    for (s in threshold)
    {
      test <- as.integer(score>=s)
      result <- c(result,1-mean(test !=ensemble_test$DIFF))
    }
    plot(threshold,result,type="l")
    cat("Meilleur seuil ", threshold[which.max(result)],"\n")
    cat("-----------------------------","\n")
    somme<-somme+threshold[which.max(result)]
  }
  
  cat("moyenne f-score",(mean(f_score)),"\n")
  cat("moyenne AUc",(mean(AUC)),"\n")
  cat("moyenne seuil",somme/15,"\n")
  
  
  # Renvoyer la moyenne des erreurs
  
  return(mean(error_rate))
}

# Exécuter la validation croisée k-fold
erreur_moyenne <- validation_kfold(data_train , k)

# Afficher le résultat
print(paste("Erreur moyenne de validation croisée k-fold:", erreur_moyenne))









##########Prediction###########



ensemble_test <- data_test
ensemble_apprentissage <- data_train

# Séparation des données et de la sortie
ensemble_apprentissage_x <- data.frame(R2=ensemble_apprentissage$R2,R14=ensemble_apprentissage$R14,R17=ensemble_apprentissage$R17,R32=ensemble_apprentissage$R32)
ensemble_test_x <- data.frame(R2=ensemble_test$R2,R14=ensemble_test$R14,R17=ensemble_test$R17,R32=ensemble_test$R32)
ensemble_apprentissage_y <- as.factor(ensemble_apprentissage$DIFF)

####### Modèle d'apprentissage (RF)


# Forêts aléatoires
rf <- randomForest(x = ensemble_apprentissage_x, y = ensemble_apprentissage_y)#ntre=500,sampsize = 100,maxnodes = 100)

# Prédiction sur l'ensemble de test
#data_test_predict <- predict(rf, newdata=ensemble_test_x, type="class")
data_test_predict <- predict(rf, newdata=ensemble_test_x, type="class")
data_test_predict_with_proba <- predict(rf, newdata=ensemble_test_x, type="prob")
seuil <-0.6
data_test_predict <- ifelse( data_test_predict_with_proba[,2] > seuil,"1","0")
print(data_test_predict)






































































