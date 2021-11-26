package dtree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

public class training {
	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder().appName("Decision Tree Model Training")
				.getOrCreate();
		
		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

		String path_training = "/user/TrainingDataset.csv";		
		String path_testing = "/user/ValidationDataset.csv";
		
		JavaRDD<String> data_training = jsc.textFile(path_training);
		JavaRDD<String> data_testing = jsc.textFile(path_testing);

		// retrieve the header
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

//		========================================================================================================
//		DECISION TREE MODEL
//		========================================================================================================
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
		double DTaccuracy = DTpredictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();

		StructType schemaAll = DataTypes
				.createStructType(new StructField[] { DataTypes.createStructField("Features", DataTypes.StringType, false),
						DataTypes.createStructField("Actual Label", DataTypes.StringType, false),
						DataTypes.createStructField("Decision Tree Prediction", DataTypes.StringType, false)});
		List<Row> rowList = new ArrayList<Row>();
		for (LabeledPoint data : test.collect()) {
			Vector features = data.getFeatures();
			Double label = data.getLabel();
			
			Double DT = DTmodel.predict(data.features());
			Row r = RowFactory.create(features.toString(), label.toString(),DT.toString());
			rowList.add(r);
		}
		
		Dataset<Row> DS = spark.sqlContext().createDataFrame(rowList, schemaAll);
		
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		DS.show();
		System.out.println("Accuracy = " + DTaccuracy + " (Decision Tree)");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");


		// Save model
		DTmodel.save(jsc.sc(), "/user/decisionTreeModel");
		
		jsc.stop();
		jsc.close();
		
	}
}
