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
