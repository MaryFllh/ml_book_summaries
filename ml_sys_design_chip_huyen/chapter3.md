# Data Engineering Fundamentals
## Data Formats
### Row-Major Versus Column-Major Format
**Row-major**: In this format, consecutive elements in a row are stored next to each other in memory. For this reason, row-major formats are good for accessing samples. The CSV format is row-major. Row-major formats are more optimal for data writes.  
**Column-major**: In this format, consecutive elements in a column are stored next to each other in memory. Because of this, column-major formats are good for accessing features from the data. The Parquet format is column-major.   

**NumPy Versus Pandas**   
Pandas is column major and therefore slow when accessing data by rows. The major order in NumPy can be specified but is row-major by default. Therefore, NumPy is much faster than Pandas in accessing rows.

### Text Versus Binary Format
Text files, e.g. CSV and JSON are in plain text and human readable. Whereas binary files, e.g. Parquet are in 0s and 1s and generally more compact. AWS suggests Parquet format because they are 2x faster to upload and take up to 6x less storage space compared to text formats.

## Data Models
Data models describe how the data is presented. For example, in a database of cars, they can be presented using the make, model, colour and its year. Alternatively, the same cars can be presented using the license plate, owner, registered addresses, etc.   
It's important to choose a representation that suits the system's needs. If we want to build a system for selling cars for example, the first data model will be a good choice. However, if we want to build a system for a law enforcement department, the second model is preferred.    
In this chapter two common data models are described:   
### Relational Model
In this model data is presented in form of relations. Each relation is a set of tuples which form rows of a table. Databases built around the relational data model are relational databases. To retrieve the data from the database, you need to use a query language. SQL is the most common query language for relational databases which is a declarative language.  

**Declarative Versus Imperative Language**   
In an imperative language like Python, you specify the action steps needed to return an output. In a declarative language like SQL however, you specify the outputs and the computer figures out the actions needed to get those outputs.

### NoSQL
NoSQL models are generally non-relational, although some NoSQL data systems are Not Only SQL and support both relational and relational models. Two common non-relational data models are document models and graph models.

**Document Model**
In this model the data is in the form of self-contained documents and the relationship between documents is rare. A document is usually a continuous string in JSON, XML or BSON (Binary JSON) format. A document is equivalent to a row in a relational table. This modal is referred as schemaless because each document can have a different set of schema. However, this is not accurate as there is no predetermined set of schema enforced on each document during writing the data, but some assumptions about schema needs to be made when reading the data.

What are the advantages of the document model?   
1. Better locality: All the information about an entity can be found in its document, as opposed to be scattered across tables that need joins to retrieve.
1. Less restriction about schema during data writes

What are the disadvantages of the document model?   
    It's less efficient to execute joins across documents compared to tables

**Graph Model**
In this model, data is also in the form of documents but the emphasis is on the relationship between them. Given this, it is faster to retrieve data based on relationships.

Unstructured data is stored in data lakes and after being processed the storage repository is called data warehouse.

## Data Storage Engines and Processing
It's important to know about different types of databases and the workloads they are optimised for in order to choose the one that best fits your needs. There are generally two types of workloads databases are optimised forl transactional processing and analytical processing.
### Transactional Processing
Any action such as watching a video, ordering an item, tweeting is considered a transaction. Transactions are inserted as they are generated, modified when something changes and deleted when no longer needed. This processing is called online transaction processing (OLTP). 
Transactional databases need to have low latency and high availabilty given that they are user facing. In this regard they can be ACID:   
1. Atomicity: guarantees that all steps in a transaction are completed successfully. If one step fails all other steps must also fail. For instance, in a ride sharing app, if the rider's payment fails, a driver should not be assigned to them.
1. Consistency: guarantees that all the inserted transactions follow predefined rules and errors in the incoming data does not create unintented consequences. For example, a user has to be validated before the transaction is accepted.
1. Isolation: guarantees that two concurrent transactions happen as if they were isolated and occuring one by one. This ensures that concurrent data read/writes do not interfer with one another. For example, two users should not be allowed to book the same driver at the same time.
1. Durability: guarantees that once a committed transaction will remain committed, even if the system fails. For instance, if a user has ordered a ride, they should still get the ride even though their phone dies.   
Transactions are processed separately; therefore, transactional databases are usually row-major.
### Analytical Processing
On the other hand, analytical databases support aggregation of values in a column across multiple rows to answer questions such as the average ride price between 5 and 6 pm last week. Analytical processing is referred to as online analytical processing (OLAP).

Optimising databases for one process or the other, was a technological limitation of the past and today with decoupling storage and compute, many databases support both.
### ETL: Extract, Transform, and Load
ETL is the process of extracting the raw data from all the data sources, validating and rejecting the malformatted data. After extracting, the data is processed and transformed to the desired structure. An example of data transformation could be standardising the range of variables across different sources, e.g. the gender column can be strings from one source and numbers in another. After transformation, based on our decision about how and the frequency, the transformed data is loaded into the target destination, e.g. database, file, data warehouse.
  <!-- <center>
    <img src="images/.jpg" width="60%" alt="etl" title="etl">
  </center> -->

### ELT: Extract, Load, and Transform
If we are not sure of the transformed data schema, we can store all the data in a data lake and leave the transformation to the application after loading the data into it. If the size of the data stored is very large, this process can become inefficient. 
## Modes of Dataflow
Usually in a real-world setting data flows from one process to another. There are three main modes of dataflow:
1. Data passing through databases
1. Data passing through services using requests, e.g. requests provided by REST and RPC APIs
1. Data passing through a real-time transport such as Apache Kafka and Amazon Kinesis

### Data Passing Through Databases
This is the easiest yet least practical mode of dataflow. In this setting, two processes will need access to the same database, e.g. process A writes to the database and process B reads that data from the database. Access to the same database may not be feasible if the two processes belong to different companies. Also, database read and writes increases latency and is not suitable for consumer-facing applications with low-latency requirements.

### Data Passing Through Services
 
