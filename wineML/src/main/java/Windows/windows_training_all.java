package Windows;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class windows_training_all {
	public static void main(String[] args) throws IOException {
		SparkSession spark = SparkSession.builder().appName("Build a DataFrame from Scratch").master("local[*]")
				.getOrCreate();

		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

		String path_training = "C:\\NJIT\\03_Fall_2021\\CS643-Sec851_Cloud_Computing\\ProgrammingAssignment2\\TrainingDataset.csv";
		String path_testing = "C:\\NJIT\\03_Fall_2021\\CS643-Sec851_Cloud_Computing\\ProgrammingAssignment2\\ValidationDataset.csv";

		JavaRDD<String> data_training = jsc.textFile(path_training);
		JavaRDD<String> data_testing = jsc.textFile(path_testing);

//		// retrieve the header
		String first_training = data_training.first();
		String first_testing = data_testing.first();

		// filter header
		JavaRDD<String> filtered_training = data_training.filter((String s) -> {
			return !s.contains(first_training);
		});
		JavaRDD<String> filtered_testing = data_testing.filter((String s) -> {
			return !s.contains(first_testing);
		});

		JavaRDD<LabeledPoint> training = filtered_training.map((String line) -> {
			String[] parts = line.split(";");
			double[] points = new double[parts.length - 1];
			for (int i = 0; i < (parts.length - 1); i++) {
				points[i] = Double.valueOf(parts[i]);
			}
			return new LabeledPoint(Double.valueOf(parts[parts.length - 1]), Vectors.dense(points));
		});

		JavaRDD<LabeledPoint> test = filtered_testing.map((String line) -> {
			String[] parts = line.split(";");
			double label = Double.valueOf(parts[parts.length - 1]);
			double[] features = new double[parts.length - 1];
			for (int i = 0; i < (parts.length - 1); i++) {
				features[i] = Double.valueOf(parts[i]);
			}

			return new LabeledPoint(label, Vectors.dense(features));
		});

//
//		StructType schema = DataTypes
//				.createStructType(new StructField[] { DataTypes.createStructField("features", DataTypes.StringType, false),
//						DataTypes.createStructField("label", DataTypes.StringType, false),
//						DataTypes.createStructField("prediction", DataTypes.StringType, false) });

		
		// ===========================LOGISTIC REGRESSION============================================
		LogisticRegressionModel LGmodel = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training.rdd());

		JavaPairRDD<Object, Object> LGpredictionAndLabels = test
				.mapToPair(p -> new Tuple2<>(LGmodel.predict(p.features()), p.label()));

		LGpredictionAndLabels.foreach(data -> {
			// System.out.println("predicted label: " + data._2 + ", actual label: " +
			// data._1);
		});
		MulticlassMetrics LGmetrics = new MulticlassMetrics(LGpredictionAndLabels.rdd());
		double LGaccuracy = LGmetrics.accuracy();
//
//		List<Row> LGrowList = new ArrayList<Row>();
//		for (LabeledPoint data : test.collect()) {
//			Vector features = data.getFeatures();
//			Double label = data.getLabel();
//			Double prediction = LGmodel.predict(data.features());
//			Row r = RowFactory.create(features.toString(), label.toString(), prediction.toString());
//			LGrowList.add(r);
//		}
//		
//		Dataset<Row> LGds = spark.sqlContext().createDataFrame(LGrowList, schema);
//		
		//============================DECISION TREE==================================================
		int numClasses = 10;
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		String impurity = "gini";
		int maxDepth = 5;
		int maxBins = 32;

		// Train a DecisionTree model for classification.
		DecisionTreeModel DTmodel = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity,
				maxDepth, maxBins);

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> DTpredictionAndLabel = test
				.mapToPair(p -> new Tuple2<>(DTmodel.predict(p.features()), p.label()));
		//double testErr = predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) test.count();
		double DTaccuracy = DTpredictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();
//
//		List<Row> DTrowList = new ArrayList<Row>();
//		for (LabeledPoint data : test.collect()) {
//			Vector features = data.getFeatures();
//			Double label = data.getLabel();
//			Double prediction = DTmodel.predict(data.features());
//			Row r = RowFactory.create(features.toString(), label.toString(), prediction.toString());
//			DTrowList.add(r);
//		}
//		
//		Dataset<Row> DTds = spark.sqlContext().createDataFrame(DTrowList, schema);
//		
		
//		//============================NAIVE BAYES==================================================		
		NaiveBayesModel NBmodel = NaiveBayes.train(training.rdd(), 1.0);
	    JavaPairRDD<Double, Double> NBpredictionAndLabel =
	      test.mapToPair(p -> new Tuple2<>(NBmodel.predict(p.features()), p.label()));
	    double NBaccuracy = NBpredictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();
//
//		List<Row> NBrowList = new ArrayList<Row>();
//		for (LabeledPoint data : test.collect()) {
//			Vector features = data.getFeatures();
//			Double label = data.getLabel();
//			Double prediction = NBmodel.predict(data.features());
//			Row r = RowFactory.create(features.toString(), label.toString(), prediction.toString());
//			NBrowList.add(r);
//		}
//		
//		Dataset<Row> NBds = spark.sqlContext().createDataFrame(NBrowList, schema);
//		
	  //===========================RANDOM FOREST===================================================
	    int numTrees = 9; 
	    String featureSubsetStrategy = "auto"; 
	    int seed = 12345;

	    RandomForestModel RFmodel = RandomForest.trainClassifier(training, numClasses,
	      categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
	      seed);

	    // Evaluate model on test instances and compute test error
	    JavaPairRDD<Double, Double> RFpredictionAndLabel =
	      test.mapToPair(p -> new Tuple2<>(RFmodel.predict(p.features()), p.label()));
	
	    double RFaccuracy = RFpredictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();
//		List<Row> RFrowList = new ArrayList<Row>();
//		for (LabeledPoint data : test.collect()) {
//			Vector features = data.getFeatures();
//			Double label = data.getLabel();
//			Double prediction = RFmodel.predict(data.features());
//			Row r = RowFactory.create(features.toString(), label.toString(), prediction.toString());
//			RFrowList.add(r);
//		}
//		
//		Dataset<Row> RFds = spark.sqlContext().createDataFrame(RFrowList, schema);
//		
		
//		System.out.println("************************************************************************");
//		System.out.println("************************************************************************");
//		System.out.println("LOGISTIC REGRESSION");
//		LGds.show();
//		System.out.println("Accuracy = " + LGaccuracy + " (Logistic Regression)");
//		System.out.println("************************************************************************");
//		System.out.println("************************************************************************");
//		System.out.println("DECISION TREE");
//		DTds.show();
//		System.out.println("Accuracy = " + DTaccuracy + " (Decision Tree)");
//		System.out.println("************************************************************************");
//		System.out.println("************************************************************************");
//		System.out.println("NAIVE BAYES");
//		NBds.show();
//		System.out.println("Accuracy = " + NBaccuracy + " (Naive Bayes)");
//		System.out.println("************************************************************************");
//		System.out.println("************************************************************************");
//		System.out.println("RANDOM FOREST");
//		RFds.show();
//		System.out.println("Accuracy = " + RFaccuracy + " (Random Forest)");
//		System.out.println("************************************************************************");
//		System.out.println("************************************************************************");
//		
		
		System.out.println("************************************************************************");
		System.out.println("************************************************************************");
		
		StructType schemaAll = DataTypes
				.createStructType(new StructField[] { DataTypes.createStructField("Features", DataTypes.StringType, false),
						DataTypes.createStructField("Actual Label", DataTypes.StringType, false),
						DataTypes.createStructField("Logistic Reg", DataTypes.StringType, false),
						DataTypes.createStructField("DecisionTree", DataTypes.StringType, false),
						DataTypes.createStructField("Naive Bayes ", DataTypes.StringType, false),
						DataTypes.createStructField("RandomForest", DataTypes.StringType, false)});

		List<Row> AllrowList = new ArrayList<Row>();
		for (LabeledPoint data : test.collect()) {
			Vector features = data.getFeatures();
			Double label = data.getLabel();
			Double LG = LGmodel.predict(data.features());
			Double DT = DTmodel.predict(data.features());
			Double NB = NBmodel.predict(data.features());
			Double RF = RFmodel.predict(data.features());
			Row r = RowFactory.create(features.toString(), label.toString(), LG.toString(), DT.toString(), NB.toString(), RF.toString());
			AllrowList.add(r);
		}
		
		Dataset<Row> Allds = spark.sqlContext().createDataFrame(AllrowList, schemaAll);
		
		Allds.show();
		System.out.println("Accuracy = " + LGaccuracy + " (Logistic Regression)");
		System.out.println("Accuracy = " + DTaccuracy + " (Decision Tree)");
		System.out.println("Accuracy = " + NBaccuracy + " (Naive Bayes)");
		System.out.println("Accuracy = " + RFaccuracy + " (Random Forest)");
		
		LGmodel.save(jsc.sc(), "file:\\\\\\C:\\NJIT\\03_Fall_2021\\CS643-Sec851_Cloud_Computing\\ProgrammingAssignment2\\trainedModel\\LGModel");
		DTmodel.save(jsc.sc(), "file:\\\\\\C:\\NJIT\\03_Fall_2021\\CS643-Sec851_Cloud_Computing\\ProgrammingAssignment2\\trainedModel\\DTModel");
		NBmodel.save(jsc.sc(), "file:\\\\\\C:\\NJIT\\03_Fall_2021\\CS643-Sec851_Cloud_Computing\\ProgrammingAssignment2\\trainedModel\\NBModel");
		RFmodel.save(jsc.sc(), "file:\\\\\\C:\\NJIT\\03_Fall_2021\\CS643-Sec851_Cloud_Computing\\ProgrammingAssignment2\\trainedModel\\RFModel");
		jsc.stop();
		jsc.close();

	}

}
