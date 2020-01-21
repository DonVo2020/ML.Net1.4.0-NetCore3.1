using System;

namespace DeepLearningsML.TensorFlowEstimator.ImageData
{
    public class ImageNetDataProbability : ImageNetData
    {
        public float Probability { get; set; }


        public void ConsoleWriteLine()
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Green;

            Console.Write($"ImagePath: {ImagePath} predicted as ");
            Console.ForegroundColor = labelColor;
            Console.Write(Label);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with probability ");
            Console.ForegroundColor = labelColor;
            Console.Write(Probability);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }
    }
}
