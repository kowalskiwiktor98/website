using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace DigitRecognition
{
    class InputData
    {
        [ColumnName("PixelValues")]
        [VectorType(64)]
        public float[] PixelValues;

        [LoadColumn(64)]
        public float Number;
    }
    class OutPutData
    {
        [ColumnName("Score")]
        public float[] Score;
    }
    class SampleMNISTData
    {
        internal static readonly InputData MNIST1 = new InputData()
        {
            PixelValues = new float[] { 0, 0, 0, 0, 14, 13, 1, 0, 0, 0, 0, 5, 16, 16, 2, 0, 0, 0, 0, 14, 16, 12, 0, 0, 0, 1, 10, 16, 16, 12, 0, 0, 0, 3, 12, 14, 16, 9, 0, 0, 0, 0, 0, 5, 16, 15, 0, 0, 0, 0, 0, 4, 16, 14, 0, 0, 0, 0, 0, 1, 13, 16, 1, 0 }
        }; //num 1


        internal static readonly InputData MNIST2 = new InputData()
        {
            PixelValues = new float[] { 0, 0, 1, 8, 15, 10, 0, 0, 0, 3, 13, 15, 14, 14, 0, 0, 0, 5, 10, 0, 10, 12, 0, 0, 0, 0, 3, 5, 15, 10, 2, 0, 0, 0, 16, 16, 16, 16, 12, 0, 0, 1, 8, 12, 14, 8, 3, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0, 0, 0, 11, 9, 0, 0, 0 }
        };//num 7

        internal static readonly InputData MNIST3 = new InputData()
        {
            PixelValues = new float[] { 0, 0, 6, 14, 4, 0, 0, 0, 0, 0, 11, 16, 10, 0, 0, 0, 0, 0, 8, 14, 16, 2, 0, 0, 0, 0, 1, 12, 12, 11, 0, 0, 0, 0, 0, 0, 0, 11, 3, 0, 0, 0, 0, 0, 0, 5, 11, 0, 0, 0, 1, 4, 4, 7, 16, 2, 0, 0, 7, 16, 16, 13, 11, 1 }
        };// num9


    }
    class Program
    {
        private static string ProjectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
        private static string TrainDataPath = ProjectDirectory + "/optdigits.tra";
        private static string TestDataPath = ProjectDirectory + "/optdigits.tes";
        private static string ModelPath = ProjectDirectory + "/Model.zip";



        static void Main(string[] args)
        {
            MLContext context = new MLContext();
            Train(context);
            TestPredictions(context);
            Console.ReadKey();
        }

        private static void Train(MLContext mlContext)
        {
            var trainData = mlContext.Data.LoadFromTextFile(path: TrainDataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );

            var testData = mlContext.Data.LoadFromTextFile(path: TestDataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Number").
                    Append(mlContext.Transforms.Concatenate("Features", nameof(InputData.PixelValues)).AppendCacheCheckpoint(mlContext));

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = trainingPipeline.Fit(trainData);

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");

            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        private static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }

        private static void TestPredictions(MLContext mlContext)
        {
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);

            var resultprediction1 = predEngine.Predict(SampleMNISTData.MNIST1);

            Console.WriteLine($"Actual: 1     Predicted probability:       zero:  {resultprediction1.Score[0]:0.####}");
            Console.WriteLine($"                                           One :  {resultprediction1.Score[1]:0.####}");
            Console.WriteLine($"                                           two:   {resultprediction1.Score[2]:0.####}");
            Console.WriteLine($"                                           three: {resultprediction1.Score[3]:0.####}");
            Console.WriteLine($"                                           four:  {resultprediction1.Score[4]:0.####}");
            Console.WriteLine($"                                           five:  {resultprediction1.Score[5]:0.####}");
            Console.WriteLine($"                                           six:   {resultprediction1.Score[6]:0.####}");
            Console.WriteLine($"                                           seven: {resultprediction1.Score[7]:0.####}");
            Console.WriteLine($"                                           eight: {resultprediction1.Score[8]:0.####}");
            Console.WriteLine($"                                           nine:  {resultprediction1.Score[9]:0.####}");
            Console.WriteLine();

            var resultprediction2 = predEngine.Predict(SampleMNISTData.MNIST2);

            Console.WriteLine($"Actual: 7     Predicted probability:       zero:  {resultprediction2.Score[0]:0.####}");
            Console.WriteLine($"                                           One :  {resultprediction2.Score[1]:0.####}");
            Console.WriteLine($"                                           two:   {resultprediction2.Score[2]:0.####}");
            Console.WriteLine($"                                           three: {resultprediction2.Score[3]:0.####}");
            Console.WriteLine($"                                           four:  {resultprediction2.Score[4]:0.####}");
            Console.WriteLine($"                                           five:  {resultprediction2.Score[5]:0.####}");
            Console.WriteLine($"                                           six:   {resultprediction2.Score[6]:0.####}");
            Console.WriteLine($"                                           seven: {resultprediction2.Score[7]:0.####}");
            Console.WriteLine($"                                           eight: {resultprediction2.Score[8]:0.####}");
            Console.WriteLine($"                                           nine:  {resultprediction2.Score[9]:0.####}");
            Console.WriteLine();

            var resultprediction3 = predEngine.Predict(SampleMNISTData.MNIST3);

            Console.WriteLine($"Actual: 9     Predicted probability:       zero:  {resultprediction3.Score[0]:0.####}");
            Console.WriteLine($"                                           One :  {resultprediction3.Score[1]:0.####}");
            Console.WriteLine($"                                           two:   {resultprediction3.Score[2]:0.####}");
            Console.WriteLine($"                                           three: {resultprediction3.Score[3]:0.####}");
            Console.WriteLine($"                                           four:  {resultprediction3.Score[4]:0.####}");
            Console.WriteLine($"                                           five:  {resultprediction3.Score[5]:0.####}");
            Console.WriteLine($"                                           six:   {resultprediction3.Score[6]:0.####}");
            Console.WriteLine($"                                           seven: {resultprediction3.Score[7]:0.####}");
            Console.WriteLine($"                                           eight: {resultprediction3.Score[8]:0.####}");
            Console.WriteLine($"                                           nine:  {resultprediction3.Score[9]:0.####}");
            Console.WriteLine();
        }
    }
}
