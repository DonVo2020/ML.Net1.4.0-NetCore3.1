using Predictions.NGrams.Text;
using System;

namespace Predictions.NGrams
{
    class Program
    {
        static void Main(string[] args)
        {
            const string kText =
                "•	Implement brand-new programs for Grants and Scholarships by working with Product Owner, BAs, and QAs teams" +
                "•	Rewrite the legacy Grants and Scholarships systems using new architecture with.Net Core, MVC, &Web API Core." +
                "	Follow Agile and Scrum Methodology with two - week sprint, grooming, tasks planning, and so on." +
                "•	Presentations / Knowledge Shares with Development Team for any new approaches and technologies." +
                 " Financial Institutions Department" +
                "•	Enhance Request / Ticket Systems and Imaging Systems by adding new features and customizations." +
                 " •	Refactor the entire of code based using Repository, Domain, Services, and Dependency Injections." +
                "•	Fix and improve UI by using JavaScript, jQuery, and MVC View Razors." +
                "•	Modify and Create SQL Stored Procedures to Support Applications." +
                "•	Enhance Imaging Systems by adding Auto Email feature and customizing UI." +
                "•	Both Request and Imaging Systems’ new features were deployed to PROD server." +
                "•	Tools: Visual Studio 2019 / 2017, TFS 2013, and SQL Management Studio 2014.";

            Text.NGrams ngs = new Text.NGrams();

            Console.WriteLine("N-Grams from 'raw' text...");
            var ngrams = ngs.GenerateNGrams(kText);

            foreach (var nGram in ngrams)
            {
                Console.WriteLine(nGram.ToString());
            }

            Console.WriteLine("========================================");

            Console.WriteLine("N-Grams from 'normalize and stop words removed' text...");
            var normalizer = new TextNormalizer();
            var normalizedText = normalizer.NormalizeText(kText);
            var stopWordsRemover = new StopWordsRemover();
            var stopWordsRemoved = stopWordsRemover.RemoveStopWords(normalizedText);

            Console.WriteLine("Normalized and Stop words removed text:");
            Console.WriteLine(stopWordsRemoved);
            Console.WriteLine("\n\n\n");
            var ngrams2 = ngs.GenerateNGrams(stopWordsRemoved);

            foreach (var nGram in ngrams2)
            {
                Console.WriteLine(nGram.ToString());
            }

            PrintEnd();
        }

        static void PrintEnd()
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("done.... press any key to continue...");
            Console.ReadKey();
        }
    }
}
