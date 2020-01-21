using DeepLearningsML.TensorFlowEstimator.Model;
using System;
using System.IO;

namespace DeepLearningsML.TensorFlowEstimator
{
    public class ImageClassificationPredict
    {
        public static void Run()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var tagsTsv = Path.Combine(assetsPath, "inputs", "data", "images_list.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "data");
            var imageClassifierZip = Path.Combine(assetsPath, "inputs", "imageClassifier.zip");

            try
            {
                var modelScorer = new ModelScorer(tagsTsv, imagesFolder, imageClassifierZip);
                modelScorer.ClassifyImages();
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
