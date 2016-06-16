package com.dp.ml.data;

import java.util.Arrays;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.GeneralizedLinearModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LassoModel;
import org.apache.spark.mllib.regression.LassoWithSGD;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.regression.RidgeRegressionModel;
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD;

import scala.Tuple2;

public class SparkLRTest {
    
    public static void main(String[] args) {
     // 屏蔽不必要的日志显示在终端上
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
        SparkConf sparkConf = new SparkConf().setAppName("Regression").setMaster("local[2]");
          JavaSparkContext sc = new JavaSparkContext(sparkConf);
          JavaRDD<String> data = sc.textFile("/export/zibian.txt");
//          JavaRDD<String> data = sc.textFile("/export/lr.txt");
          
//          val train_data = sample_data.map( v =>
//              Array.tabulate[Double](field_cnt)(
//                  i => zscore(v._2(i),sample_mean(i),sample_stddev(i))
//              )
//          ).cache
          
          JavaRDD<LabeledPoint> parsedData = data.map(line -> {
              String[] parts = line.split(",");
              double[] ds = Arrays.stream(parts[1].split(" "))
                    .mapToDouble(Double::parseDouble)
                    .toArray();
              return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(ds));
          }).cache();
          
//          stepSize: 迭代步长，默认为1.0  更新器第 t 步的步长等于 stepSize / sqrt(t)
//          numIterations: 迭代次数，默认为100
//          regParam: 正则化参数，默认值为0.0
//          miniBatchFraction: 每次迭代参与计算的样本比例，默认为1.0
//          gradient：LogisticGradient()，Logistic梯度下降；
//          updater：SquaredL2Updater()，正则化，L2范数；
//          optimizer：GradientDescent(gradient, updater)，梯度下降最优化计算。
          double stepSize=1;
          int numIterations = 1;
          double miniBatchFraction=1;
          LinearRegressionModel model = LinearRegressionWithSGD.train(parsedData.rdd(), numIterations,stepSize,miniBatchFraction);
          RidgeRegressionModel model1 = RidgeRegressionWithSGD.train(parsedData.rdd(), numIterations);
          LassoModel model2 = LassoWithSGD.train(parsedData.rdd(), numIterations);
        
          print(parsedData, model);
          print(parsedData, model1);
          print(parsedData, model2);
        
          //预测一条新数据方法
//          double[] d = new double[]{1.0, 1.0, 2.0, 1.0, 3.0, -1.0, 1.0, -2.0};
//          double[] d = new double[]{38406, 290, 233,10,2};
          double[] d = new double[]{763, 632, 932, 763, 721, 562, 345, 623};
          Vector v = Vectors.dense(d);
          System.out.println("model predict:"+model.predict(v));
          System.out.println("model1 predict:"+model1.predict(v));
          System.out.println("model2 predict:"+model2.predict(v));
    }
    public static void print(JavaRDD<LabeledPoint> parsedData, GeneralizedLinearModel model) {
        JavaPairRDD<Double, Double> valuesAndPreds = parsedData.mapToPair(point -> {
            double prediction = model.predict(point.features()); //用模型预测训练数据
            return new Tuple2<>(point.label(), prediction);
        });
     
        Double MSE = valuesAndPreds.mapToDouble((Tuple2<Double, Double> t) -> Math.pow(t._1() - t._2(), 2)).mean(); //计算预测值与实际值差值的平方值的均值
        System.out.println(model.getClass().getName() + " training Mean Squared Error = " + MSE);
    }
}
