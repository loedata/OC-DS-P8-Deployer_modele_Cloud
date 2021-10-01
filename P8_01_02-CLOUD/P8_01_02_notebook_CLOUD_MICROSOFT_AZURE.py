# Databricks notebook source
# MAGIC %md # P8 - Déployer un modèle dans le cloud

# COMMAND ----------

# MAGIC %md
# MAGIC <img style="padding-left:130px;" src = '/files/tables/resources/images/entreprise_fruits.png' />

# COMMAND ----------

# MAGIC %md # P8_01_02 - CLOUD - Utilisation de Microsoft Azure

# COMMAND ----------

# MAGIC %md Ce notebook traite de du chargement du jeu de données des images, du pré-processing, de la réduction de dimension et d'une mini classification pour des nouvelles images en utilisant l'outils **Microsoft Databricks Azure** pour la partie "**compute**" (calculs distribués) et un **container blob de stockage** "**Data Lake Storage**".

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Introduction
# MAGIC 
# MAGIC *****
# MAGIC **Mission**
# MAGIC *****
# MAGIC **Développer dans un environnement Big Data une première chaîne de traitement des données qui comprendra le preprocessing et une étape de réduction de dimension** pour une startup Fruits! de l'AgriTech pour mettre à disposition du grand public une application mobile qui permettrait aux utilisateurs de prendre en photo un fruit et d'obtenir des informations sur ce fruit.
# MAGIC 
# MAGIC *****
# MAGIC **Contraintes**
# MAGIC *****
# MAGIC - Le volume de données va augmenter très rapidement après la livraison de ce projet.
# MAGIC - Développer des scripts en Pyspark.
# MAGIC - Utiliser le cloud AWS ou autre (Microsoft Azure sera utilisé pour ce projet) pour profiter d’une architecture Big Data. 
# MAGIC 
# MAGIC *****
# MAGIC **Sources**
# MAGIC *****
# MAGIC - [Jeu de données](https://www.kaggle.com/moltean/fruits) : constitué des images de fruits et des labels associés, qui pourra servir de point de départ pour construire une partie de la chaîne de traitement des données.
# MAGIC 
# MAGIC **Information sur le jeu de données :**
# MAGIC ***
# MAGIC 
# MAGIC - Nombre **total d'images** : **90483**.
# MAGIC - Jeu de données **train set** : **67692** images (1 fruit ou 1 légumes par image).
# MAGIC - Jeu de données **test set** : **22688**  images (1 fruit ou 1 légumes par image).
# MAGIC - Nombre de **classes** : 131 (fruits ou légumes).
# MAGIC - **Taille** des images : 100x100 pixels.
# MAGIC - **Format du nom de fichier** : 
# MAGIC   - **imageindex100.jpg** (par exemple 32100.jpg),
# MAGIC   - ou **rimageindex100.jpg** (par exemple r32100.jpg),
# MAGIC   - ou **r2imageindex100.jpg**,
# MAGIC   - ou **r3imageindex100.jpg**. 
# MAGIC      - ou "r" signifie que le fruit a subi une rotation,
# MAGIC      - "r2" signifie que le fruit a été tourné autour du 3ème axe,
# MAGIC      - "100" vient de la taille de l'image (100x100 pixels).
# MAGIC - Exemples de classe : Apples (different varieties: Crimson Snow, Golden, Golden-Red, Granny Smith, Pink Lady, Red, Red Delicious), Apricot, Avocado, Avocado ripe, Banana (Yellow, Red, Lady Finger), Beetroot Red, Blueberry, Cactus fruit, Cantaloupe (2 varieties), Carambula, Cauliflower, Cherry (different varieties, Rainier), Cherry Wax (Yellow, Red, Black), Chestnut, Clementine, Cocos, Corn (with husk), Cucumber (ripened), Dates, Eggplant, Fig, Ginger Root, Granadilla, Grape (Blue, Pink, White (different varieties)), Grapefruit (Pink, White), Guava, Hazelnut, Huckleberry, Kiwi, Kaki, Kohlrabi, Kumsquats, Lemon (normal, Meyer), Lime, Lychee, Mandarine, Mango (Green, Red), Mangostan, Maracuja, Melon Piel de Sapo, Mulberry, Nectarine (Regular, Flat), Nut (Forest, Pecan), Onion (Red, White), Orange, Papaya, Passion fruit, Peach (different varieties), Pepino, Pear (different varieties, Abate, Forelle, Kaiser, Monster, Red, Stone, Williams), Pepper (Red, Green, Orange, Yellow), Physalis (normal, with Husk), Pineapple (normal, Mini), Pitahaya Red, Plum (different varieties), Pomegranate, Pomelo Sweetie, Potato (Red, Sweet, White), Quince, Rambutan, Raspberry, Redcurrant, Salak, Strawberry (normal, Wedge), Tamarillo, Tangelo, Tomato (different varieties, Maroon, Cherry Red, Yellow, not ripened, Heart), Walnut, Watermelon.

# COMMAND ----------

# Chargement des librairies
import datetime
import io
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Visualisation
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Pyspark
import pyspark
from pyspark.sql.functions import element_at, split, col, pandas_udf, PandasUDFType, udf
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession

# Tensorflow Keras
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Gestion des images
import PIL
from PIL import Image

# Taches ML
from pyspark.ml.image import ImageSchema

# Réduction de dimension - PCA
from pyspark.ml.feature import PCA
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector

# Modélisation
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Matrice de confusion
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
import itertools

%matplotlib inline

# Versions
print('Version des librairies utilisées :')
print('Python        : ' + sys.version)
print('tensorflow    : ' + tf.__version__)
print('pyspark       : ' + pyspark.__version__)
print('PIL           : ' + PIL.__version__)
print('Numpy         : ' + np.__version__)
print('Pandas        : ' + pd.__version__)
print('Matplotlib    : ' + matplotlib.__version__)
print('Seaborn       : ' + sns.__version__)
print('Sklearn       : ' + sklearn.__version__)
now = datetime.now().isoformat()
print('Lancé le      : ' + now)

# COMMAND ----------

# MAGIC %md ## 2. Préparation des données du train set

# COMMAND ----------

# MAGIC %md ### 2.1. Jeu de données train set - au format "binaryFile"

# COMMAND ----------

# MAGIC %md #### 2.2.1. Connection du container blob de stockage à Azure Databricks 

# COMMAND ----------

dbutils.fs.unmount("/mnt/p8-cloud")

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://p8-cloud@p8cloud.blob.core.windows.net",
  mount_point = "/mnt/p8-cloud",
  extra_configs = {"fs.azure.account.key.p8cloud.blob.core.windows.net":"eO/iSSs9527scMebXetNiaApnpamHEq0G95gUYs3cBIjU3F4ZCUyx9xtzk9b05Aa8gqtL7kVXS94lTcWBFyBEQ=="})

# COMMAND ----------

# MAGIC %md #### 2.2.2. Chargement du jeu de données train set - au format "binaryFile"

# COMMAND ----------

# MAGIC %md **Recommandation databricks** : 
# MAGIC 
# MAGIC Databricks recommande d’utiliser la source de données de fichier binaire pour charger des données d’image dans le tableau Spark en tant qu’octets bruts.
# MAGIC 
# MAGIC [Source](https://docs.microsoft.com/fr-fr/azure/databricks/data/data-sources/image)
# MAGIC 
# MAGIC ***Note***:
# MAGIC ***
# MAGIC Ce format sera ainsi utilisé pour la mise en production via Microsoft Azure.

# COMMAND ----------

# Chemin de stockage des images du jeu de données
path_train_set = "/mnt/p8-cloud/resources/data/train-set/*/*"

# COMMAND ----------

# Chargement des images du train set au format "binaryFile"
df_binary_train = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load(path_train_set)

# COMMAND ----------

# Schéma ?
df_binary_train.printSchema()

# COMMAND ----------

# Nombre d'images?
df_binary_train.count()

# COMMAND ----------

# Visualisation des 20 premières images
df_binary_train.show()

# COMMAND ----------

# MAGIC %md #### 2.2.2. Labellisation - extraction de la classe de l'image

# COMMAND ----------

# MAGIC %md 
# MAGIC - La classe de l'image traitée est définie dans le nom du répertoire de l'image.
# MAGIC - Exemple : répertoire : dbfs:/FileStore/tables/resources/data/train-set/**Onion-Red**/43_100.jpg ==> **classe='Onion-Red'**.

# COMMAND ----------

# Ajout dans la colonne Classe pour chaque image traitée de l'avant dernier
# élément du nom du répertoire de stockage de l'image==>df_binary_train["path"]
df_binary_train = df_binary_train.withColumn("Classe", element_at(split(df_binary_train["path"], "/"), -2))

# COMMAND ----------

# Schéma ?
df_binary_train.printSchema()

# COMMAND ----------

# Visualisation des 20 premières images avec la classe
df_binary_train.show()

# COMMAND ----------

# MAGIC %md ## 3. Extraction des features importantes pour chaque image

# COMMAND ----------

# MAGIC %md 
# MAGIC - Comme vu lors du projet 6, l'extraction des features par transfert learning donne des résultats plus performants que les méthodes anciennes (ORB, SIFT). 
# MAGIC - Nous allons donc extraire les features les plus importantes pour la classification de nos images en utilisant un modèle **[InceptionV3](https://www.researchgate.net/figure/Schematic-diagram-of-the-Inception-v3-model-based-on-convolutional-neural-networks_fig3_337200783)** de deep learning pré-entrainé sur de la classification d'images.
# MAGIC - Comme le but de ce projet n'est pas d'effectuer la classification; La dernière couche (softmax), qui effectue la classification, est supprimée à l'aide du paramètre (include_top=False). Cela nous permettra de choisir un modèle de classification adapté à nos classes.

# COMMAND ----------

# MAGIC %md ### 3.1. Préparation du dataframe de travail

# COMMAND ----------

# MAGIC %md 
# MAGIC Le dataframe de travail sera composé des colonnes utiles à partir du dataframe des images binaires :
# MAGIC - le répertoire de stockage de l'image (colonne path),
# MAGIC - le label (colonne Classe) de chaque image,
# MAGIC - les features les plus importantes ajoutées après exécution du modèle (étape 3.3.).

# COMMAND ----------

df_images = df_binary_train.select("path", "Classe")
df_images.show()

# COMMAND ----------

# MAGIC %md ### 3.2. Préparation du modèle InceptionV3

# COMMAND ----------

# MAGIC %md Utilisation de la technique de transfert learning pour extraire les features de chaque image avec le modèle **[InceptionV3](https://keras.io/api/applications/inceptionv3/)** de la librairie Keras de tensorflow avec l'aide des recommandations de databricks sur l'utilisation du transfert learning.

# COMMAND ----------

# MAGIC %md ***Note : Recommandation de databricks***
# MAGIC ***
# MAGIC - Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# MAGIC If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
# MAGIC `spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")`
# MAGIC - Pour les modèles de taille modérée (< 1 Go, c'est le cas de notre projet), une bonne pratique consiste à télécharger le modèle vers le pilote Spark, puis à diffuser les poids aux travailleurs. Ce carnet de notes utilise cette approche.
# MAGIC `bc_model_weights = sc.broadcast(model.get_weights())`
# MAGIC `model.set_weights(bc_model_weights.value)`
# MAGIC 
# MAGIC [Source](https://docs.databricks.com/_static/notebooks/deep-learning/deep-learning-transfer-learning-keras.html)

# COMMAND ----------

# Instanciation du modèle
model = InceptionV3(
        include_top=False,  # Couche softmax de classification supprimée
        weights='imagenet',  # Poids pré-entraînés sur Imagenet
        input_shape=(100,100,3), # Image de taille 100x100 en couleur (channel=3)
        pooling='max' # Utilisation du max de pooling
)

# COMMAND ----------

# Description des caractéristiques du modèle
model.summary()

# COMMAND ----------

# MAGIC %md Le vecteur des features est de dimension de chaque image à la dimensions (1, 1, 2048)

# COMMAND ----------

# MAGIC %md ### 3.3. Extraction des features pour chaque image

# COMMAND ----------

# MAGIC %md [Source_databricks](https://docs.databricks.com/_static/notebooks/deep-learning/deep-learning-transfer-learning-keras.html)

# COMMAND ----------

# MAGIC %md #### 3.3.1. Fonctions utiles à l'extraction des features

# COMMAND ----------

# MAGIC %md **Préparation du modèle**
# MAGIC ***

# COMMAND ----------

# Instanciation du modèle
model = InceptionV3(
        include_top=False,  # Couche softmax de classification supprimée
        weights='imagenet',  # Poids pré-entraînés sur Imagenet
        input_shape=(100,100,3), # Image de taille 100x100 en couleur (channel=3)
        pooling='max' # Utilisation du max de pooling
)

# COMMAND ----------

# Permettre aux workers Spark d'accéder aux poids utilisés par le modèle
bc_model_weights = spark.sparkContext.broadcast(model.get_weights())

# COMMAND ----------

def model_fn():
  """
  Renvoie un modèle Inception3 avec la couche supérieure supprimée et les poids pré-entraînés sur imagenet diffusés.
  """
  model = InceptionV3(
        include_top=False,  # Couche softmax de classification supprimée
        weights='imagenet',  # Poids pré-entraînés sur Imagenet
#         input_shape=(100,100,3), # Image de taille 100x100 en couleur (channel=3)
        pooling='max' # Utilisation du max de pooling
  )
  model.set_weights(bc_model_weights.value)
  
  return model

# COMMAND ----------

# MAGIC %md **Fonction de redimensionnement de l'image**
# MAGIC ***
# MAGIC Les images à transmettre en entrée de InceptionV3 doivent entre de dimension (299,299, 3)

# COMMAND ----------

# Redimensionnement des images en 299x299
def preprocess(content):
    """
    Prétraite les octets de l'image brute pour la prédiction.
    param : content : objet image, obligatoire
    return : image redimensionnée en Array
    """
    # lecture + redimension (299x299) pour Xception
    img = PIL.Image.open(io.BytesIO(content)).resize([299, 299])
    # transforme l'image en Array     
    arr = img_to_array(img)
    return preprocess_input(arr)

# COMMAND ----------

# MAGIC %md **Extraction des features par le modèle dans un vecteur**
# MAGIC ***

# COMMAND ----------

# Extraction des features par le modèle dans un vecteur
def featurize_series(model, content_series):
  """
  Featurise une pd.Series d'images brutes en utilisant le modèle d'entrée.
  param : 
    model : modèle à utiliser pour l'extraction, obligatoire.
    content_series : image redimensionnée (299, 299, 3) en Array
  :return: les features importantes de l'image en pd.Series.
  """
  input = np.stack(content_series.map(preprocess))
  # Prédiction du modèle
  preds = model.predict(input)
  # Pour certaines couches, les caractéristiques de sortie seront des tenseurs multidimensionnels.
  # Nous aplatissons les tenseurs de caractéristiques en vecteurs pour faciliter le stockage dans
  # les DataFrames de Spark.
  output = [p.flatten() for p in preds]
  
  return pd.Series(output)

# COMMAND ----------

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
  '''
  Cette méthode est un Scalar Iterator pandas UDF enveloppant notre fonction de featurisation.
  Le décorateur spécifie que cette méthode renvoie une colonne Spark DataFrame de type ArrayType(FloatType).
  
  :param content_series_iter : Cet argument est un itérateur sur des lots de données, où chaque lot est une série pandas de données d'image.
  '''
  # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
  # for multiple data batches.  This amortizes the overhead of loading big models.
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)

# COMMAND ----------

# MAGIC %md #### 3.3.2. Extraction des features pour chaque image du dataframe

# COMMAND ----------

# Les UDF de Pandas sur de grands enregistrements (par exemple, de très grandes images) peuvent rencontrer des erreurs de type Out Of Memory (OOM).
# Si vous rencontrez de telles erreurs dans la cellule ci-dessous, essayez de réduire la taille du lot Arrow via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# COMMAND ----------

# Nous pouvons maintenant exécuter la featurisation sur l'ensemble de notre DataFrame Spark.
# REMARQUE : Cela peut prendre beaucoup de temps (environ 10 minutes) car il applique un grand modèle à l'ensemble des données.
features_df = df_binary_train.repartition(16).select(col("path"), col('Classe'), featurize_udf("content").alias("features"))

# COMMAND ----------

# 4484 images?
features_df.count()

# COMMAND ----------

# MAGIC %md ### 3.4. Réduction de dimension - Principal Component Analysis

# COMMAND ----------

# MAGIC %md #### 3.4.1. Recherche meilleur nombre de composante

# COMMAND ----------

# MAGIC %md **Préparation des données**

# COMMAND ----------

def preprocess_pca(dataframe):
  '''
     Préparation des données :
     - transformation en vecteur dense
     - standardisation
     param : dataframe : dataframe d'images
     return : dataframe avec features vecteur dense standardisé
  '''
  
  # Préparation des données - conversion des données images en vecteur dense
  transform_vecteur_dense = udf(lambda r: Vectors.dense(r), VectorUDT())
  dataframe = dataframe.withColumn('features_vectors', transform_vecteur_dense('features'))
  
  # Standardisation obligatoire pour PCA
  scaler_std = StandardScaler(inputCol="features_vectors", outputCol="features_scaled", withStd=True, withMean=True)
  model_std = scaler_std.fit(dataframe)
  # Mise à l'échelle
  dataframe = model_std.transform(dataframe)
  
  return dataframe

# COMMAND ----------

# MAGIC %md **Recherche du nombre de composante expliquant 95% de la variance**

# COMMAND ----------

def recherche_nb_composante(dataframe, nb_comp=400):
    '''
       Recherche d nombre de composante expliquant 95% de la variance
       param : dataframe : dataframe d'images
       return : k nombre de composante expliquant 95% de la variance totale
    '''
    
    pca = PCA(k = nb_comp,
              inputCol="features_scaled", 
              outputCol="features_pca")
 
    model_pca = pca.fit(dataframe)
    variance = model_pca.explainedVariance
 
    # visuel
    plt.plot(np.arange(len(variance)) + 1, variance.cumsum(), c="red", marker='o')
    plt.xlabel("Nb composantes")
    plt.ylabel("% variance")
    plt.show(block=False)
 
    def nb_comp ():
      for i in range(500):
          a = variance.cumsum()[i]
          if a >= 0.95:
              print("{} composantes principales expliquent au moins 95% de la variance totale".format(i))
              break
      return i
 
    k=nb_comp()
  
    return k


# COMMAND ----------

# Pré-processing (vecteur dense, standardisation)
df_pca = preprocess_pca(features_df)

# COMMAND ----------

# Nombre de composante expliquant 95% de la variance
n_components = recherche_nb_composante(df_pca)

# COMMAND ----------

# 325 composantes expliquent plus de 90% de la variance
n_components = 325

# COMMAND ----------

# MAGIC %md #### 3.4.2. Réduction de dimension PCA

# COMMAND ----------

# Entrainement de l'algorithme
pca = PCA(k=n_components, inputCol='features_scaled', outputCol='vectors_pca')
model_pca = pca.fit(df_pca)

# COMMAND ----------

# Transformation des images sur les k premières composantes
df_reduit = model_pca.transform(df_pca)

# COMMAND ----------

# Visualisation du dataframe réduit
df_reduit.show()

# COMMAND ----------

# MAGIC %md #### 3.4.3. Sauvegarde des données

# COMMAND ----------

# MAGIC %md Finalement, on sauvegarde les données pré-traitées et réduites au format parquet.

# COMMAND ----------

# Sauvegarde des données
df_reduit.write.mode("overwrite").parquet("/mnt/p8-cloud/resources/output/resultats_features_parquet")

# COMMAND ----------

# MAGIC %md ## 4. Test de classification

# COMMAND ----------

# MAGIC %md ### 4.1. Préparation des données

# COMMAND ----------

# MAGIC %md **Seed**

# COMMAND ----------

# Nombre aléatoire pour la reproductibilité des résultats
seed = 21

# COMMAND ----------

# MAGIC %md **Dataframe de travail**

# COMMAND ----------

# Chargement du dataframe sauvegardé en parquet
parquetFiles = "/mnt/p8-cloud/resources/output/resultats_features_parquet/"

df_reduit = spark.read.parquet(parquetFiles)

# COMMAND ----------

# Conservation de la classe de l'image et des vecteurs pca
data = df_reduit[["Classe", "vectors_pca"]]

# COMMAND ----------

data.show(5)

# COMMAND ----------

# MAGIC %md **Encodage de la variable cible**

# COMMAND ----------

# Encodage de la variable cible : la classe de l'image acceptable par le modèle
indexer = StringIndexer(inputCol="Classe", outputCol="Classe_index")

# Fit the indexer to learn Classe/index pairs
indexerModel = indexer.fit(data)

# Append a new column with the index
data = indexerModel.transform(data)

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md **Découpage du jeu du train set en jeux d'entraînement et de validation**

# COMMAND ----------

# data splitting
(train_data, valid_data) = data.randomSplit([0.8, 0.2])

# COMMAND ----------

print("Nbre élément train_data : " + str(train_data.count()))
print("Nbre élément valid_data : " + str(valid_data.count()))

# COMMAND ----------

display(train_data.head(3))

# COMMAND ----------

# MAGIC %md ### 4.2. Modélisation Logistic Regression

# COMMAND ----------

# MAGIC %md ***Modélisation Régression Logistique***

# COMMAND ----------

# MAGIC %md [Source](https://spark.apache.org/docs/latest/ml-classification-regression.html#multinomial-logistic-regression)

# COMMAND ----------

# MAGIC %md ***Entraînement du modèle***

# COMMAND ----------

# Instanciation du modèle.
lr = LogisticRegression(labelCol="Classe_index", featuresCol="vectors_pca",
                        maxIter=5)

# Entraînement du modèle
lr_model = lr.fit(train_data)

# COMMAND ----------

# MAGIC %md ***Prédictions***

# COMMAND ----------

# Make predictions.
lr_predictions = lr_model.transform(valid_data)

# Select example rows to display.
lr_predictions.select("prediction", "Classe_index").show(5)

# COMMAND ----------

# MAGIC %md ***Évaluation du modèle***

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="Classe_index", predictionCol="prediction", metricName="accuracy")
lr_accuracy = evaluator.evaluate(lr_predictions)
print("Test Error = %g" % (1.0 - lr_accuracy))
print("Accuracy = %g " % lr_accuracy)

# COMMAND ----------

# MAGIC %md ***Informations sur le modèle***

# COMMAND ----------

# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lr_model.coefficientMatrix))
print("Intercept: " + str(lr_model.interceptVector))

trainingSummary = lr_model.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))

print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

# COMMAND ----------

# MAGIC %md ### 4.3. Modélisation Decision Tree Classifier

# COMMAND ----------

# MAGIC %md ***Modélisation Decision Tree Classifier***

# COMMAND ----------

# MAGIC %md [Source](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier)

# COMMAND ----------

# MAGIC %md ***Entraînement du modèle***

# COMMAND ----------

# Instanciation du modèle.
dtc = DecisionTreeClassifier(labelCol="Classe_index", featuresCol="vectors_pca",
                             seed=seed)

# Entraînement du modèle
dtc_model = dtc.fit(train_data)

# COMMAND ----------

# MAGIC %md ***Prédictions***

# COMMAND ----------

# Make predictions.
dtc_predictions = dtc_model.transform(valid_data)

# Select example rows to display.
dtc_predictions.select("prediction", "Classe_index").show(5)

# COMMAND ----------

# MAGIC %md ***Évaluation du modèle***

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="Classe_index", predictionCol="prediction", metricName="accuracy")
dtc_accuracy = evaluator.evaluate(dtc_predictions)
print("Test Error = %g" % (1.0 - dtc_accuracy))
print("Accuracy = %g " % dtc_accuracy)

# COMMAND ----------

# MAGIC %md ***Informations sur le modèle***

# COMMAND ----------

print(dtc_model.toDebugString)

# COMMAND ----------

# MAGIC %md ### 4.4. Modélisation Random Forest Classifier

# COMMAND ----------

# MAGIC %md ***Modélisation Random Forest Classifier***

# COMMAND ----------

# MAGIC %md [Source](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier)

# COMMAND ----------

# MAGIC %md ***Entraînement du modèle***

# COMMAND ----------

# Instanciation du modèle.
rf = RandomForestClassifier(labelCol="Classe_index", featuresCol="vectors_pca", numTrees=20,
                             seed=seed)

# Entraînement du modèle
rf_model = rf.fit(train_data)

# COMMAND ----------

# MAGIC %md ***Prédictions***

# COMMAND ----------

# Make predictions.
rf_predictions = rf_model.transform(valid_data)

# Select example rows to display.
rf_predictions.select("prediction", "Classe_index").show(5)

# COMMAND ----------

# MAGIC %md ***Évaluation du modèle***

# COMMAND ----------

# Select (prediction, true label) and compute test error
rf_evaluator = MulticlassClassificationEvaluator(
    labelCol="Classe_index", predictionCol="prediction", metricName="accuracy")
rf_accuracy = rf_evaluator.evaluate(rf_predictions)
print("Test Error = %g" % (1.0 - rf_accuracy))
print("Accuracy = %g " % rf_accuracy)

# COMMAND ----------

# MAGIC %md ***Matrice de confusion***

# COMMAND ----------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), 
                                  range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# COMMAND ----------

y_true = rf_predictions.select("Classe_Index")
y_true = y_true.toPandas()

y_pred = rf_predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred, labels=range(10))

# COMMAND ----------

sns.set_style("darkgrid")

plt.figure(figsize=(7,7))
plt.grid(False)

# call pre defined function
plot_confusion_matrix(cnf_matrix, classes=range(10))

# COMMAND ----------

target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(y_true, y_pred, target_names = target_names))

# COMMAND ----------

# MAGIC %md ***Informations sur le modèle***

# COMMAND ----------

print(rf_model.toDebugString)
