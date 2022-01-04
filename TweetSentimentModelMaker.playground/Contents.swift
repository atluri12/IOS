import Cocoa
import CreateML

let data = try MLDataTable(
    contentsOf:URL(fileURLWithPath:"/Users/ramamohanraoveeramachaneni/Desktop/Sentiment.csv"))

let(trainingData,testingData) = data.randomSplit(by:0.80,seed:5)

let sentimentClassfier = try MLTextClassifier(trainingData: trainingData, textColumn: "filtered_text", labelColumn: "Sentiment_f")
let evaluationMetrics = sentimentClassfier.evaluation(on: testingData,textColumn: "text",labelColumn: "class")

let evaluationAccuracy = (1.0-evaluationMetrics.classificationError)*100

//let metadata = MLModelMetadata(author: "Maharshi", shortDescription: "Model to classify twitter sentiment", version: "1.0")
//try sentimentClassfier.write(to: URL(fileURLWithPath: "/Users/ramamohanraoveeramachaneni/Downloads/TwitterSentimentClassfier.mlmodel"))
