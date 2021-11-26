package randomforest;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class prediction {
	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder().appName("Random Forest Prediction").master("local[*]")
				.getOrCreate();

		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

		String path_testing = "/app/target/data/TestingDataset.csv";				
		
		JavaRDD<String> data_testing = jsc.textFile(path_testing);

		// retrieve the header
		String first_testing = data_testing.first();

		// filter header
		JavaRDD<String> filtered_testing = data_testing.filter((String s) -> {
			return !s.contains(first_testing);
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

		// ############################RANDOM FOREST######################################
		RandomForestModel RFmodel = RandomForestModel.load(jsc.sc(), "/app/target/models/randomForestModel");

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> RFpredictionAndLabel = test
				.mapToPair(p -> new Tuple2<>(RFmodel.predict(p.features()), p.label()));
		double RFaccuracy = RFpredictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();

		
		StructType schemaAll = DataTypes
				.createStructType(new StructField[] { DataTypes.createStructField("Features", DataTypes.StringType, false),
						DataTypes.createStructField("Actual Label", DataTypes.StringType, false),
						DataTypes.createStructField("RandomForest", DataTypes.StringType, false)});

		List<Row> AllrowList = new ArrayList<Row>();
		for (LabeledPoint data : test.collect()) {
			Vector features = data.getFeatures();
			Double label = data.getLabel();
			Double RF = RFmodel.predict(data.features());
			Row r = RowFactory.create(features.toString(), label.toString(), RF.toString());
			AllrowList.add(r);
		}
		
		Dataset<Row> Allds = spark.sqlContext().createDataFrame(AllrowList, schemaAll);

		
		Allds.show();
		System.out.println("Accuracy = " + RFaccuracy + " (Random Forest)");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		System.out.println("");
		
		
		jsc.stop();
		jsc.close();
	}
}
