# Examples for productionizing H2O

## Training

Model training is taken out in python, see ```src``` folder.

## Prediction

Predictions are taken out with Java and the H2O REST API.

To run the java example:
```
cd models
javac -cp h2o-genmodel.jar -J-Xms2g MojoWrapper.java
java -cp .:h2o-genmodel.jar:gson-2.6.2.jar MojoWrapper '{"Sepal.Length":"5.1","Sepal.Width":"3.5", "Petal.Length":"1.4", "Petal.Width":"0.2"}'
```

To predict with the H2O REST API:
```
python src/train_and_predict.py
```
