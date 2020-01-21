using System;

namespace DeepLearningsML.TensorFlowEstimator
{
    class Program
    {
        static void Main(string[] args)
        {
            ImageClassificationTrain.Run();
            ImageClassificationPredict.Run();
        }
    }
}
