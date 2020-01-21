using System.Runtime.Serialization;

namespace MulticlassClassificationsML.BankTransactionWithJson.Models
{
    [DataContract]
    public class TransactionData
    {
        [DataMember(Name = "id")]
        public string ID { get; set; }

        [DataMember(Name = "information")]
        public string Information { get; set; }

        [DataMember(Name = "category")]
        public string Category { get; set; }

        [DataMember(Name = "creditDebitIndicator")]
        public string CreditDebitIndicator { get; set; }
    }
}
