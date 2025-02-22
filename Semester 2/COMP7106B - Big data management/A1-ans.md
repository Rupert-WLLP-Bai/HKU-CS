# (a) Relational Algebra

1. **Find the titles of the Novel books which use Chinese as the language.**

$$
\pi_{title}\Big(\sigma_{\mathit{genre}="Novel" \land \mathit{language}="Chinese"}(Book)\Big)
$$

2. **Retrieve the names of female customers who have borrowed Novel books and are due for return on 01-01-2025.**
$$
\pi_{name}\Big(
     \sigma_{\mathit{gender}="Ms."}(Customer)
     \ \Join_{Customer.cID = Borrow.cID}\ 
     \sigma_{\mathit{dueDate}="01-01-2025"}(Borrow)
     \ \Join_{Borrow.bID = Book.bID}\ 
     \sigma_{\mathit{genre}="Novel"}(Book)
   \Big)
$$
---

# (b) SQL Queries

1. **Display the distinct genres of books borrowed by Mr. customers whose ages are between 40 and 60.**

   ```sql
   SELECT DISTINCT B.genre
   FROM Customer AS C
   JOIN Borrow AS BR ON C.cID = BR.cID
   JOIN Book AS B ON BR.bID = B.bID
   WHERE C.gender = 'Mr.'
     AND C.age BETWEEN 40 AND 60;

2. **For each genre of books, display the genre and the average age of customers.**

    ```sql
    SELECT B.genre, AVG(C.age) AS avg_age
    FROM Customer AS C
    JOIN Borrow AS BR ON C.cID = BR.cID
    JOIN Book AS B ON BR.bID = B.bID
    GROUP BY B.genre;
    ```
