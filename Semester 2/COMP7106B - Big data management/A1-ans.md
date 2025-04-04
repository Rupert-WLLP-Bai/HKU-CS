## 1. (a) Relational Algebra

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

## 1. (b) SQL Queries

1. **Display the distinct genres of books borrowed by Mr. customers whose ages are between 40 and 60.**

   ```sql
   SELECT DISTINCT B.genre
   FROM Customer AS C
   JOIN Borrow AS BR ON C.cID = BR.cID
   JOIN Book AS B ON BR.bID = B.bID
   WHERE C.gender = 'Mr.'
     AND C.age BETWEEN 40 AND 60;
   ```

2. **For each genre of books, display the genre and the average age of customers.**

    ```sql
    SELECT B.genre, AVG(C.age) AS avg_age
    FROM Customer AS C
    JOIN Borrow AS BR ON C.cID = BR.cID
    JOIN Book AS B ON BR.bID = B.bID
    GROUP BY B.genre;
    ```

## 2

![image-20250302153644407](./Q2.jpg)

| Node | Q                                                                              | oNN  | dist(q, oNN) | rating       |
| ---- | ------------------------------------------------------------------------------ | ---- | ------------ | ------------ |
| Root | M1(1), M2(2), M3(4)                                                            | null | inf          |              |
| M1   | m2($\sqrt2$), m1(2), M2(2), M3(4)                                              | null | inf          |              |
| m2   | a($\sqrt2$), m1(2)，M2(2), b($\sqrt5$), c(2$\sqrt2$), M3(4)                    | null | inf          |              |
| a    | m1(2), M2(2), b($\sqrt5$), c(2$\sqrt2$), M3(4)                                 | null | inf          | 6 $\not > 6$ |
| m1   | M2(2), b($\sqrt5$), e($\sqrt5$), c(2$\sqrt2$), e(4), M3(4)                     | null | inf          |              |
| M2   | m3(2), b($\sqrt5$), e($\sqrt5$), c(2$\sqrt2$), e(4), m4(4), M3(4)              | null | inf          |              |
| m3   | f(2), b($\sqrt5$), e($\sqrt5$), c(2$\sqrt2$), g($\sqrt10$), e(4), m4(4), M3(4) | null | inf          |              |
| f    | b($\sqrt5$), e($\sqrt5$), c(2$\sqrt2$), g($\sqrt10$), e(4), m4(4), M3(4)       | f    | 2            | 7 > 6        |

### Result

oNN = f 

dist(q, oNN) = 2

Node Accessed = 1 + 3 + 4 + 7 = 15