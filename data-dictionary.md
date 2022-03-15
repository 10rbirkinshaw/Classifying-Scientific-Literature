| Column Name | Description | Type | Count | Used in Modelling  |     
|  ------  | -----|----|----|-----|
| date        | Date the article was published | str | 246574 non-null | No |
| title | Title of the scientific article | str | 246801 non-null | Yes | 
| abstract        | Abstract of the scientific article | str | 246801 non-null | Yes |
| journal       | Journal the scientific article was published in | str | 246574 non-null | No |
| tags   | Keyword tags associated with content in the scientific article | str | 246800 non-null  | Yes | 
| authors        | List of authors that wrote the scientific article | str | 241780 non-null | No | 
| art_type     | Type of scientific article (e.g. review, clinical trial) | str | 67553 non-null | No |
| art_language | Language of article publication - used to ensure modelling was done on only english articles | str | 31784  non-null | No |
| year | Feature engineered from date column | int64 | 161720 non-null | Yes |
| label | Feature engineered target label assigning the journal to a scientific domain | str | 161720 non-null | Target for classification models |
