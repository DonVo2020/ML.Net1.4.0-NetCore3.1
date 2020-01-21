using Microsoft.ML.Data;

namespace MulticlassClassificationsML.BankTransactionWithJson.Models
{
    public class BankTransactionPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category;
    }
}
