import pandas as pd

data = {
    "Name": ["John Doe", "Alice Smith", "Bob Johnson", "Emily Davis"],
    "Email": ["john.doe@example.com", "alice.smith@domain.com", "bob.johnson@work.com", "emily.davis@company.org"],
    "Phone": ["123-456-7890", "987-654-3210", "555-555-5555", "666-666-6666"],
    "Address": ["123 Main St", "456 Elm St", "789 Oak St", "101 Pine St"],
    "Date_of_Birth": ["1990-01-01", "1985-02-10", "1992-03-15", "1989-04-20"],
}

df = pd.DataFrame(data)
df.to_csv("personal_data.csv", index=False)