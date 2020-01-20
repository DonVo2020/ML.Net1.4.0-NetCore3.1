using System;

namespace BinaryClassificationML.CreditCardFraudDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            TrainnerMain.Run();
            PredictorMain.Run();
        }
    }
}
