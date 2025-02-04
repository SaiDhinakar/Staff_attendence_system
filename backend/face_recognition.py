import sqlite3
import json
from datetime import datetime

class AttendanceDB:
    def __init__(self, db_name="attendance.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        """Initialize the database and create tables if they don't exist."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Employee Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employee (
                Emp_id INTEGER PRIMARY KEY AUTOINCREMENT,
                Emp_name TEXT NOT NULL,
                Department TEXT NOT NULL
            )
        """)

        # Attendance Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Date TEXT NOT NULL,
                Emp_id INTEGER NOT NULL,
                Emp_name TEXT NOT NULL,
                Time TEXT NOT NULL,
                FOREIGN KEY (Emp_id) REFERENCES employee (Emp_id)
            )
        """)

        conn.commit()
        conn.close()

    def add_employee(self, emp_name, department):
        """Add a new employee to the database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO employee (Emp_name, Department) VALUES (?, ?)", (emp_name, department))
        conn.commit()
        conn.close()

    def log_attendance(self, emp_id):
        """Log an employee's attendance with timestamp."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        now = datetime.now().strftime("%I:%M %p")  # Format time as 09:00 AM
        today_date = datetime.now().strftime("%Y-%m-%d")

        # Check if the employee exists
        cursor.execute("SELECT Emp_name FROM employee WHERE Emp_id = ?", (emp_id,))
        row = cursor.fetchone()
        if not row:
            print(f"Employee ID {emp_id} not found!")
            conn.close()
            return

        emp_name = row[0]

        # Check if an entry exists for today
        cursor.execute("SELECT Time FROM attendance WHERE Emp_id = ? AND Date = ?", (emp_id, today_date))
        row = cursor.fetchone()

        if row:
            timestamps = row[0].split(",")  # Convert stored string to list
            last_time = datetime.strptime(timestamps[-1], "%I:%M %p")
            now_time = datetime.strptime(now, "%I:%M %p")

            # Ensure a minimum of 5 minutes gap
            if (now_time - last_time).total_seconds() >= 300:
                timestamps.append(now)
                cursor.execute("UPDATE attendance SET Time = ? WHERE Emp_id = ? AND Date = ?",
                               (",".join(timestamps), emp_id, today_date))
        else:
            cursor.execute("INSERT INTO attendance (Date, Emp_id, Emp_name, Time) VALUES (?, ?, ?, ?)",
                           (today_date, emp_id, emp_name, now))

        conn.commit()
        conn.close()

    def get_attendance(self):
        """Retrieve attendance records."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance")
        records = cursor.fetchall()
        conn.close()
        return records

    def get_employees(self):
        """Retrieve all employees."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employee")
        records = cursor.fetchall()
        conn.close()
        return records


# Example Usage:
db = AttendanceDB()

# Add Employees (Only Run Once)
db.add_employee("Alice Johnson", "HR")
db.add_employee("Bob Smith", "IT")
db.add_employee("Charlie Davis", "Finance")
db.add_employee("David Brown", "Marketing")
db.add_employee("Emma Wilson", "Engineering")

# Log Attendance
db.log_attendance(1)  # Alice Johnson
db.log_attendance(2)  # Bob Smith

# Print Tables
print("Attendance Records:", db.get_attendance())
print("Employees:", db.get_employees())
