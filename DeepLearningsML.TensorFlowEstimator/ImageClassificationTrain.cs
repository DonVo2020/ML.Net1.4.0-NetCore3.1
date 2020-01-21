using DeepLearningsML.TensorFlowEstimator.Model;
using System;
using System.IO;

namespace DeepLearningsML.TensorFlowEstimator
{
    public class ImageClassificationTrain
    {
        public static void Run()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var tagsTsv = Path.Combine(assetsPath, "inputs", "data", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "data");
            var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            var imageClassifierZip = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");

            try
            {
                var modelBuilder = new ModelBuilder(tagsTsv, imagesFolder, inceptionPb, imageClassifierZip);
                modelBuilder.BuildAndTrain();
            }
            catch (Exception ex)
            {
                ConsoleHelpers.ConsoleWriteException(ex.Message);
            }

            ConsoleHelpers.ConsolePressAnyKey();
        }

        private static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
