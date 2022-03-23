import sc.utils.generate_report as report
import sc.utils.analysis as analysis
import pymongo as mg

if __name__ == "__main__":
    myclient = mg.MongoClient("mongodb://localhost:2022")
    
