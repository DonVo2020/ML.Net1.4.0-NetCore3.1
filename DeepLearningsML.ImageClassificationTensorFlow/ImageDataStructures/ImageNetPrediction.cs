using DeepLearningsML.ImageClassificationTensorFlow.ModelScorer;
using Microsoft.ML.Data;

namespace DeepLearningsML.ImageClassificationTensorFlow.ImageDataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName(TFModelScorer.InceptionSettings.outputTensorName)]
        public float[] PredictedLabels;
    }
}
