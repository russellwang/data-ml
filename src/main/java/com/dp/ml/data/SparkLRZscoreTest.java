package com.dp.ml.data;

import java.io.File;
import java.util.Arrays;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.GeneralizedLinearModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;

public class SparkLRZscoreTest {
    
    public static void main(String[] args) {
     // 屏蔽不必要的日志显示在终端上
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
        SparkConf sparkConf = new SparkConf().setAppName("Regression").setMaster("local[2]");
          JavaSparkContext sc = new JavaSparkContext(sparkConf);
          JavaRDD<String> data = sc.textFile("/export/shopgmvblank.txt");

          Normalizer normalizer1 = new Normalizer();
          JavaRDD<LabeledPoint> zscoreData = data.map(line -> {
              String[] parts = line.split(" ");
              double[] ds = Arrays.stream(parts)
                      .mapToDouble(Double::parseDouble)
                      .toArray();
              Vector v=normalizer1.transform(Vectors.dense(ds));
              System.out.println(v);
              double[] tmp=new double[v.size()-1];
              for(int i=1;i<v.size();i++) {
                  tmp[i-1]=v.apply(i);
              }
              return new LabeledPoint(v.apply(0), Vectors.dense(tmp));
          }).cache();
          
          JavaRDD<LabeledPoint> parsedData = data.map(line -> {
              String[] parts = line.split(" ");
              double[] ds = Arrays.stream(parts)
                    .mapToDouble(Double::parseDouble)
                    .toArray();
              double[] tmp=new double[ds.length-1];
              for(int i=1;i<ds.length;i++) {
                  tmp[i-1]=ds[i];
              }
              return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(tmp));
          }).cache();
          
//          stepSize: 迭代步长，默认为1.0  更新器第 t 步的步长等于 stepSize / sqrt(t)
//          numIterations: 迭代次数，默认为100
//          regParam: 正则化参数，默认值为0.0
//          miniBatchFraction: 每次迭代参与计算的样本比例，默认为1.0
//          gradient：LogisticGradient()，Logistic梯度下降；
//          updater：SquaredL2Updater()，正则化，L2范数；
//          optimizer：GradientDescent(gradient, updater)，梯度下降最优化计算。
          
//          for(int i=1700;i<=1710;i++) {
//              for(int j=50;j<=50;j++) {
//                  LinearRegressionModel model = LinearRegressionWithSGD.train(zscoreData.rdd(), i,j,1.0);
//                  print(i,j,zscoreData, model);
//              }
//          }
          
          int stepSize=50;
          int numIterations = 1700;
          double miniBatchFraction=1;
          LinearRegressionModel model = LinearRegressionWithSGD.train(zscoreData.rdd(), numIterations,stepSize,miniBatchFraction);
          print(numIterations,stepSize,zscoreData, model);
          System.out.println("model weight:"+model.weights());
          
//          //预测一条新数据方法
//          double[] d=new double[]{5575, 49, 41, 10, 0};
//          Vector v = Vectors.dense(d);
//          System.out.println("model predict:"+model.predict(v));
          
          
          JavaPairRDD<Double, Double> valuesAndPreds = parsedData.mapToPair(point -> {
              double prediction = model.predict(point.features()); //用模型预测训练数据
              return new Tuple2<>(prediction,point.label());
          });
          File file=new File("/export/predict1.txt");
          if(file.exists()) {
              file.delete();
          }
          valuesAndPreds.saveAsTextFile("/export/predict1.txt");
    }
    
    public static void print(int i,int j,JavaRDD<LabeledPoint> parsedData, GeneralizedLinearModel model) {
        JavaPairRDD<Double, Double> valuesAndPreds = parsedData.mapToPair(point -> {
            double prediction = model.predict(point.features()); //用模型预测训练数据
            return new Tuple2<>(point.label(), prediction);
        });
     
        Double MSE = valuesAndPreds.mapToDouble((Tuple2<Double, Double> t) -> Math.pow(t._1() - t._2(), 2)).mean(); //计算预测值与实际值差值的平方值的均值
        System.out.println(i+"  "+j+"  "+model.getClass().getName() + " training Mean Squared Error = " + MSE);
    }
}
