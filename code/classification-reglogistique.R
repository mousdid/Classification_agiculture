
# Adresse du dossier où vous travaillez
setwd("C:/Users/as/Desktop/TP_R/Projet/Code")
# Packages utilisés dans la suite
library(class)
# Packages utilisés dans la suite
library(caret)
library(ROCR)
library(writexl)

# Supprimer toutes les variables
rm(list=ls(all=TRUE))
# Supprimer tous les graphiques déjà présents
graphics.off()
# Lecture des données d’apprentissage
load("../Data/Projets/farm_data_train.rda");
load("../Data/Projets/farm_data_test.rda");



#data_train_x <- data.frame(R2=data_train$R2,R14=data_train$R14,R17=data_train$R17,R32=data_train$R32)

#######kfold



# Nombre de plis (k)
k <- 15


# Fonction pour effectuer la validation croisée k-fold
validation_kfold <- function(data_train, k) {
  set.seed(123)  # Pour la reproductibilité
  indices <- sample(1:k, nrow(data_train), replace = TRUE)
  
  error_rate <- numeric(k)
  f_score <- numeric(k)
  AUC <-numeric(k)
  somme <- 0.
  
  
  for (i in 1:k) {
    ensemble_test <- data_train[indices == i, ]
    ensemble_apprentissage <- data_train[indices != i, ]
    
    # Séparation des données et de la sortie
    ensemble_apprentissage_x <- data.frame(R2=ensemble_apprentissage$R2,R14=ensemble_apprentissage$R14,R17=ensemble_apprentissage$R17,R32=ensemble_apprentissage$R32)
    ensemble_test_x <- data.frame(R2=ensemble_test$R2,R14=ensemble_test$R14,R17=ensemble_test$R17,R32=ensemble_test$R32)
    ensemble_apprentissage_y <- as.factor(ensemble_apprentissage$DIFF)
    
    ####### Modèle d'apprentissage (RL)
    
    
    # Régression logistique
    glm_train <- glm(DIFF~., data = data_train, family=binomial())
    
    # Prédiction sur l'ensemble de test
   glm_train_predict  <- predict(glm_train, newdata=ensemble_test_x, type="response")
   seuil <-0.4
   print("------------------------------------------------------")
   data_test_predict <- (glm_train_predict > seuil)
   data_test_predict_with_proba <-glm_train_predict
   
   
   #seuil <-0.8
   #data_test_predict <- ifelse( data_test_predict_with_proba =="0",ifelse( (data_test_predict_with_proba) >seuil ,"0","1"),ifelse(1- (data_test_predict_with_proba) >seuil ,"0","1"))
   
   #data_test_predict_with_proba <- glm_train_predict
   #seuil <-0.4
   #data_test_predict <- ifelse( data_test_predict_with_proba > seuil,"1","0")
   print(glm_train_predict)
    

    print (ensemble_test$DIFF)
    
    
    
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
    
    cat("F-score = ",2 * TPR * PPV / (TPR+PPV),"\n")
    
    f_score[i]=2 * TPR * PPV / (TPR+PPV)
    
    # k plus proches voisins avec les probas
    
    #data_test_predict_with_proba <- predict(glm_train, newdata=ensemble_test_x, type="response")
   
    print(glm_train_predict)
    # Calcul du score
    
    
    data_test_predict_with_proba <- glm_train_predict
    score <- data_test_predict_with_proba
    
    
    #score <- ifelse(data_test_predict == "0", 1-score, score)
    
    
    
    pred_lda <- prediction(score, ensemble_test$DIFF)
    perf <- performance(pred_lda, "tpr", "fpr")
    plot(perf,colorize=TRUE)
    par(new=T)
    plot(c(0,1),c(0,1),type="l",ann=FALSE)
    
    # Aire sous la courbe
    AUC[i] <- performance(pred_lda, "auc")@y.values[[1]]
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
    somme <-somme+ threshold[which.max(result)]
  }
  
  cat("moyenne f-score",(mean(f_score)),"\n")
  cat("moyenne AUC",(mean(AUC)),"\n")
  cat("moyennesuil",somme/15,"\n")
  
  
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


####### Modèle d'apprentissage (RL)


# Régression logistique
glm_train <- glm(DIFF~., data = data_train, family=binomial())

# Prédiction sur l'ensemble de test
glm_train_predict  <- predict(glm_train, newdata=ensemble_test_x, type="response")
seuil <-0.4
print("------------------------------------------------------")
data_test_predict <- (glm_train_predict > seuil)+1-1
data_test_predict_with_proba <-glm_train_predict
print(data_test_predict)

# Spécifier le chemin du fichier Excel
chemin_fichier_excel <- "C:/Users/as/Desktop/TP_R/Projet/Code/prediction_finale.xlsx"

# Écrire dans le fichier Excel
data_test_predict  <- cbind(data_test,data_test_predict)
data_test_predict <- as.data.frame(data_test_predict)
write_xlsx(data_test_predict, chemin_fichier_excel)



























































































































