//cd C:\spark-2.4.6-bin-hadoop2.7\bin
//spark shell --driver-memory 6g

///////////////////////////////////
////////// Decision Tree //////////
///////////////////////////////////
//의사결정나무는 데이터 내의 이상치(Outlier)에 잘 휘둘리지 않는다.
//전처리/정규화 과정을 걸치지 않고도 라는 유형과 다른 척도의 데이터를 다룰 수 있다.
//장점 : 직관적으로 이해할 수 있고 추론할 수 있다.

//데이터 : http://bit.ly/1KiJRfg

//////////////////////
//// 6. DATA 준비 ////
//////////////////////
//// 01 ////
val dataWithoutHeader = spark.read.
    option("inferSchema",true).
    option("header",false).
    csv("./Chap4_data/covtype.data")
//One-Hot Encoding된 열 다수 존재.

//// 02 ////
val colNames=Seq(
    "Elevation","Aspect","Slope",
    "Horizontal_Distance_To_Hydrology","Vertical__Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am","Hillshade_Noon","Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
    )++(
        (0 until 4).map(i=>s"Wilderness_Area_$i")
    )++(
        (0 until 40).map(i=>s"Soil_Type_$i")
    )++Seq("Cover_Type")

val data= dataWithoutHeader.toDF(colNames:_*).
    withColumn("Cover_Type",$"Cover_Type".cast("double"))

data.head

////////////////////////////////
//// 7. First Decision Tree ////
////////////////////////////////
//// 01 ////
// data의 90% : train, 10% : test에 이용
val Array(trainData, testData)= data.randomSplit(Array(0.9,0.1))
trainData.cache()
testData.cache()

//// 02 ////
import org.apache.spark.ml.feature.VectorAssembler
/*
VectorAssembler를 이용하여
Cover_Type을 제외한 변수들을 이용하여 특징벡터를 생성한다.
VectorAssembler는 MLlib 파이프라인 API 중 Transformer에 해당한다.
하나의 DF을 또 다른 DF로 변환하고,
이러한 변환들을 묶어 하나의 파이프라인으로 구성한다.
*/
val inputCols = trainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().
    setInputCols(inputCols).
    setOutputCol("featureVector")

val assembledTrainData = assembler.transform(trainData)
assembledTrainData.select("featureVector").show(truncate=false)

//// 03 ////
import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random

val classifier = new DecisionTreeClassifier().
    setSeed(Random.nextLong()).
    setLabelCol("Cover_Type").
    setFeaturesCol("featureVector").
    setPredictionCol("prediction")
//반드시, 입력할 "특징벡터"열, 예측할 "목표값"의 이름을 설정한다.
//이후 재사용을 위해 예측을 저장할 열도 설정한다.

val model= classifier.fit(assembledTrainData)
println(model.toDebugString)

//// 04 ////
//변수(입력특징) 중요도 확인
model.featureImportances.toArray.zip(inputCols).
    sorted.reverse.foreach(println)

//Elevation(고도)의 변수 중요도 : 약 0.82 정도인데 반해
//2위가 0.042정도로 결과를 좌지우지 하는 것은 Elevation(고도)인 것을 확인할 수 있다.

//대략적 예측 결과 확인
val predictions = model.transform(assembledTrainData)
predictions.select("Cover_Type", "prediction","probability").
    show(truncate=false)
//Probability : 각 결과가 맞을 확률에 대한 모델의 추정치

//MulticlassClassificationEvaluator를 이용해 모델의 예측 품질을 평가할 '정확도'를 구한다.
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluator = new MulticlassClassificationEvaluator().
    setLabelCol("Cover_Type").
    setPredictionCol("prediction")

evaluator.setMetricName("accuracy").evaluate(predictions)
//Accuracy : 0.70 (이번 분석에서는 Accuracy 사용)
evaluator.setMetricName("f1").evaluate(predictions)
//F1 : 0.69

//// 05 ////
//Confusion Matrix
// 7 x 7 
import org.apache.spark.mllib.evaluation.MulticlassMetrics
val predictionRDD = predictions.
    select("prediction","Cover_Type").
    as[(Double,Double)]. //->dataset
    rdd//dataset->rdd
val multiclassMetrics= new MulticlassMetrics(predictionRDD)
multiclassMetrics.confusionMatrix
/*
res10: org.apache.spark.mllib.linalg.Matrix =
143480.0  43965.0   72.0     0.0     44.0   6.0    3040.0
63316.0   187618.0  3015.0   42.0    324.0  45.0   511.0
0.0       4439.0    27293.0  345.0   51.0   100.0  0.0
0.0       6.0       1329.0   1109.0  0.0    0.0    0.0
95.0      7762.0    282.0    1.0     408.0  0.0    0.0
0.0       4776.0    10261.0  125.0   15.0   460.0  0.0
10356.0   79.0      0.0      0.0     0.0    0.0    8086.0

=> (i,i)값, 즉 ↘ 대각선 상의 값이 크니 어느정도 잘 예측한 것으로 판단됨.
=> 그러나 (5,5)를 보면 꽤나 미심쩍은 부분도 있다.
*/

//// 06 ////
val confusionMatrix=predictions.
    groupBy("Cover_Type").
    pivot("prediction",(1 to 7)).
    count().
    na.fill(0.0).
    orderBy("Cover_Type")

confusionMatrix.show()
/* 좀 더 보기 편하게 표현 (스파크 SQL 이용)
+----------+------+------+-----+----+---+---+----+
|Cover_Type|     1|     2|    3|   4|  5|  6|   7|
+----------+------+------+-----+----+---+---+----+
|       1.0|143480| 43965|   72|   0| 44|  6|3040|
|       2.0| 63316|187618| 3015|  42|324| 45| 511|
|       3.0|     0|  4439|27293| 345| 51|100|   0|
|       4.0|     0|     6| 1329|1109|  0|  0|   0|
|       5.0|    95|  7762|  282|   1|408|  0|   0|
|       6.0|     0|  4776|10261| 125| 15|460|   0|
|       7.0| 10356|    79|    0|   0|  0|  0|8086|
+----------+------+------+-----+----+---+---+----+
*/

//// 07 ////
/*
Accuracy가 70%면 괜찮아 보일 수도 있지만, 명확하게 판단 불가.
만약 Train set의 30%가 대표수종이 1이라 하고,
Test set의 40%가 대표수종이 1이라고 한다면,
대표수종 1은 30% * 40% = 0.3 * 0.4 = 0.12 = 12% 만큼
전체 정확도에 기여한다.
이를 계산해본다.
*/
import org.apache.spark.sql.DataFrame

def classProbabilities(data: DataFrame): Array[Double]={
    val total = data.count()
    data.groupBy("Cover_Type").count().
    select("count").as[Double].
    map(_/total).collect()
}

val trainPriorProbabilities=classProbabilities(trainData)
val testPriorProbabilities=classProbabilities(testData)
val accuracy = trainPriorProbabilities.zip(testPriorProbabilities).map{
    case(trainProb, testProb)=>trainProb * testProb
}.sum
//accuracy: Double = 0.3767894798576171
//이를 통해 무작위로 추측했을 때는 38% 정도 정확하다고 한다.
//이에 비하면 앞서 얻은 70%의 정확도는 좋은 결과로 판단된다.
//그러나 이는 하이퍼 파라미터 수정없이 진행한 것으로 수정 후엔
//정확도를 더욱 개선할 수 있을 것이다.


/////////////////////////////////////////////
//// 8. Hyper Parameter on Decision Tree ////
/////////////////////////////////////////////
/*
하이퍼 파라미터 선택 : AUC 아닌 다중 범주의 정확도를 지표로 사용
1) Maximum Depth(최대 깊이)
2) Maximum Bins(최대 통 수) : 커질수록 처리속도 느려짐 but 최적화도니 결정규칙 찾을 수 있다.
3) Impurity Measure(불순도) : Gini 계수, Entropy를 이용
    3-1) Gini 계수 : 부분집합내에서 무작위로 선택한 표본을 무작위로 선택해 분류했을 때 틀릴 확률
                (스파크에서는 Gini계수로 기본적으로 설정)
    3-2) Entropy : 부분집합 속 목표값의 집합에 불확실성이 얼마나 많이 포함되어 있는가를 표현
4) Minimum Inforamtion Gain(최소 정보 획득량) : 후보 결정 규칙을 대상으로 최소 정보 획득량을 강요하거나 불순도를 낮추기 위해 도입된 하이퍼 파라미터.
 -> 과적합 방지
*/

////////////////////////////////////
//// 9. Tuning on Decision Tree ////
////////////////////////////////////
/*
1) 어떤 불순도 측정 방식을 사용하면 정확도가 높은가
2) 어느 정도의 최대 깊이와 통의 수가 적절한가

-> 이에 대한 값의 조합을 시험 후, 결과 확인
-> 이것들을 캡슐화하는 파이프라인을 설정해야 함

-> VectorAssembler와 DecisionTreeClassifier를 생성하고
-> 이 2개의 Transformer를 연결하면, 
-> 이러한 두 작업으을 하나의 작업으로 표현하는 Pipeline 객체 생성됨 
*/

//// 01 ////
import org.apache.spark.ml.Pipeline

val inputCols = trainData.columns.filter(_ != "Cover_Type")

val assembler = new VectorAssembler().
    setInputCols(inputCols).
    setOutputCol("featureVector")

val classifier = new DecisionTreeClassifier().
    setSeed(Random.nextLong()).
    setLabelCol("Cover_Type").
    setFeaturesCol("featureVector").
    setPredictionCol("prediction")

val pipeline = new Pipeline().setStages(Array(assembler,classifier))

//// 02 ////
//PramGridBuilder를 사용하여 테스트해야 하는 하이퍼 파라미터의 조합을 정의할 수 있다.
//최적의 하이퍼 파라미터를 선정하는 데 사용할 평가지표를 정의하는 순서라고 할 수 있다.
import org.apache.spark.ml.tuning.ParamGridBuilder

val paramGrid = new ParamGridBuilder().
    addGrid(classifier.impurity, Seq("gini","entropy")).
    addGrid(classifier.maxDepth, Seq(1,20)).
    addGrid(classifier.maxBins, Seq(40,300)).
    addGrid(classifier.minInfoGain, Seq(0.0,0.05)).
    build()
    
val multiclassEval = new MulticlassClassificationEvaluator().
    setLabelCol("Cover_Type").
    setPredictionCol("Prediction").
    setMetricName("accuracy")
//4^2 = 16 개의 모델 : 4개의 하이퍼 파라미터당 2개씩 값 적용
//평가지표 : 다중분류 정확도

////03 ////
//k- cross k cross validation을 수행하는데, CrossValidator 사용할 수 있지만,
//비용이 k배로 증가하므로, 데이터 양이 충분히 많은 경우, 추가로 얻는 이득이 많지 않다.
//그러므로 여기서는 TrainValidationSplit을 사용한다.
import org.apache.spark.ml.tuning.TrainValidationSplit

val validator =  new TrainValidationSplit().
    setSeed(Random.nextLong()).
    setEstimator(pipeline).
    setEvaluator(multiclassEval).
    setEstimatorParamMaps(paramGrid).
    setTrainRatio(0.9)//Train set:validation set = 9:1
//validation set : train data에 적합시킨 파라미터 평가
//test set : validation data에 적하십시킨 하이퍼 파리미터 평가
val validatorModel = validator.fit(trainData)
//validator가 반환한 결과는 찾아낸 최적의 모델을 담고 있다.
//이는 최적의 파이프라인을 찾아 반환하는 것이다.

//// 04 ////
import org.apache.spark.ml.PipelineModel

val bestModel = validatorModel.bestModel
bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap
/*
결과
- 불순도 측정에는 Entropy가 좋다.
- maxDepth : 20나 1이나 별 차이가 없음.
- maxBins : 40개 정도면 충분하다고 해석 가능.
= minInfoGain : 최소로 설정한 작은 값에서 더 좋다는 결과는
    이 모델은 과적합보다는 과소적합에 취약하다는 것을 말함
*/

//// 05 ////
//하이퍼 파라미터의 각 조합에 대한 각 모델의 정확도
val validatorModel = validator.fit(trainData)

val paramsAndMetrics = validatorModel.validationMetrics.
    zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)

paramsAndMetrics.foreach{ case (metric, params)=>
    println(metric)
    println(params)
    println()
}


//validation set에서 달성한 정확도
//test set에서 달성한 정확도
validatorModel.validationMetrics.max
multiclassEval.evaluate(bestModel.transform(testData))
//(bestModel은 온전한 파이프라인)
//for validation set: 0.9138483377774368
//for test set: 0.9139978718291971




///////////////////////////////
//// 10. 범주형 변수 다루기 ////
//////////////////////////////
//// 01 ////
import org.apache.spark.sql.functions._

def unencodeOneHot(data: DataFrame): DataFrame={
    val wildernessCols = (0 until 4).map(i=>s"Wilderness_Area_$i").toArray
    
    val wildernessAssembler = new VectorAssembler().
        setInputCols(wildernessCols).
        setOutputCol("wilderness")

    val unhotUDF = udf((vec: Vector)=> vec.toArray.indexOf(1.0).toDouble)

    val withWilderness = wildernessAssembler.transform(data).
        drop(wildernessCols:_*).
        withColumn("wilderness",unhotUDF($"wilderness"))

    val soilCols = (0 until 40).map(i=>$s"Soil_Type_$i").toArray
    
    val soilAssembler = new VectorAssembler().
        setInputCols(soilCols).
        setOutputCol("soil")

    soilAssembler.transform(withWilderness).
        drop(soilCols:_*).
        withColumn("soil",unhotUDF($"soil"))
}
/*
VectorAssembler()는 
1)4개의 황야 유형, 
2) 40개의 토양 유형
을 각각 하나의 벡터로 결합하는데 사용한다.
-> 이 벡터의 값은 하나만 1이고 나머지는 모두 0이다.
-> 이를 위해 자체 UDF를 정의한 것.
-> 이를 통해 MLlib에 입력으로 사용할 수 있도록 숫자로 바꿀 수 있음.
*/

//// 02 ////
/*
VectorIndexor
MLlib에서는 각 열에 대한 추가적인 메타정보를 저장할 수 있다.
(메타정보로 저장된 상세내용은 숨겨져 있지만, 범주형 값을 인코딩한 열인지 혹은 고유한 값이 몇개나 있는지 등의 정보가 들어있음.)
이러한 메타정보를 추가할 때, VectorIndexor를 사용한다.
VectorIndexor는 입력값을 적절하게 라벨링된 범주형 변수열로 바꿔준다.
이러한 것을 이제 파이프라인에 추가한다.
*/
import org.apache.spark.ml.feature.VectorIndexor

val inputCols=unencTrainData.columns.filter(_!="Cover_Type")
val assembler = new VectorAssembler().
    setInputCols(inputCols).
    setOutputCol("featureVector")

val indexer = new VectorIndexor().
    setMaxCategories(40).//토양의 값이 40가지이므로 그 이상이어야 함.
    setInputCol("featureVector").
    setOutputCol("indexedVector")

val classifier = new DecisionTreeClassifier().
    setSeed(Random.nextLong()).
    setLabelCol("Cover_Type").
    setFeaturesCol("indexedVector").
    setPredictionCol("prediction")

val pipeline = new Pipeline().setStages(Array(assembler,indexer,classifier))



