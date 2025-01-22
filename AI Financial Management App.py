import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
import sqlite3
from datetime import datetime

class PersonalFinanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Finance Management App")
        self.root.geometry("900x600")
        self.root.config(bg="white")

        self.setup_database()
        
        self.income_data = []
        self.expense_data = []
        self.load_data_from_db()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.home_tab = tk.Frame(self.notebook, bg="white")
        self.predict_tab = tk.Frame(self.notebook, bg="white")
        self.visualize_tab = tk.Frame(self.notebook, bg="white")
        self.input_tab = tk.Frame(self.notebook, bg="white")
        self.manage_tab = tk.Frame(self.notebook, bg="white")

        self.notebook.add(self.home_tab, text="Home")
        self.notebook.add(self.predict_tab, text="Prediction")
        self.notebook.add(self.visualize_tab, text="Visualization")
        self.notebook.add(self.input_tab, text="Transaction Input")
        self.notebook.add(self.manage_tab, text="Manage Transactions")

        # Home Tab
        tk.Label(self.home_tab, text="Personal Finance Management System", font=("Arial", 20, "bold"), bg="white").pack(pady=20)

        tk.Button(self.home_tab, text="Track Your Expenses", font=("Arial", 12), command=self.track_expenses).pack(pady=10)
        tk.Button(self.home_tab, text="See Money Tips", font=("Arial", 12), command=self.see_money_tips).pack(pady=10)

        # Prediction Tab
        tk.Label(self.predict_tab, text="Predict Next Month's Spending", font=("Arial", 16, "bold"), bg="white").pack(pady=20)

        tk.Label(self.predict_tab, text="Enter Last 5 Months' Income and Expenditure:", font=("Arial", 12), bg="white").pack(pady=5)

        self.income_entries = []
        self.expense_entries = []

        for i in range(5):
            frame = tk.Frame(self.predict_tab, bg="white")
            frame.pack(pady=5)
            tk.Label(frame, text=f"Month {i+1} Income:", font=("Arial", 10), bg="white").pack(side=tk.LEFT, padx=5)
            income_entry = tk.Entry(frame, font=("Arial", 10))
            income_entry.pack(side=tk.LEFT, padx=5)
            self.income_entries.append(income_entry)

            tk.Label(frame, text=f"Month {i+1} Expense:", font=("Arial", 10), bg="white").pack(side=tk.LEFT, padx=5)
            expense_entry = tk.Entry(frame, font=("Arial", 10))
            expense_entry.pack(side=tk.LEFT, padx=5)
            self.expense_entries.append(expense_entry)

        tk.Button(self.predict_tab, text="Predict", font=("Arial", 12), command=self.predict_next_month_spending).pack(pady=10)
        self.prediction_label = tk.Label(self.predict_tab, text="", font=("Arial", 12), bg="white")
        self.prediction_label.pack(pady=10)

        # Visualization Tab
        tk.Label(self.visualize_tab, text="Income and Spending Visualization", font=("Arial", 16, "bold"), bg="white").pack(pady=20)
        tk.Button(self.visualize_tab, text="Show Visualization", font=("Arial", 12), command=self.visualize_data).pack(pady=10)

        # Transaction Input Tab
        tk.Label(self.input_tab, text="Add Your Transactions", font=("Arial", 16, "bold"), bg="white").pack(pady=20)

        tk.Label(self.input_tab, text="Income ($):", font=("Arial", 12), bg="white").pack(pady=5)
        self.new_income_entry = tk.Entry(self.input_tab, font=("Arial", 12))
        self.new_income_entry.pack(pady=5)

        tk.Label(self.input_tab, text="Spending ($):", font=("Arial", 12), bg="white").pack(pady=5)
        self.new_spending_entry = tk.Entry(self.input_tab, font=("Arial", 12))
        self.new_spending_entry.pack(pady=5)

        tk.Button(self.input_tab, text="Add Transaction", font=("Arial", 12), command=self.add_transaction).pack(pady=10)

        plt.style.use('bmh')
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = None

        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

#Setup
        self.setup_visualization_tab()
        self.setup_prediction_tab()
        self.setup_manage_transactions_tab()
        self.setup_input_tab()

    def setup_database(self):
        """Create database and tables if they don't exist"""
        try:
            self.conn = sqlite3.connect('finance_data.db')
            self.cursor = self.conn.cursor()
            
            # Create transactions table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    month INTEGER PRIMARY KEY,
                    income REAL,
                    spending REAL,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
        except Exception as e:
            messagebox.showerror("Database Error", f"Error setting up database: {str(e)}")

    def load_data_from_db(self):
        """Load existing data from database"""
        try:
            self.cursor.execute('SELECT month, income, spending FROM transactions ORDER BY month')
            data = self.cursor.execute('SELECT month, income, spending FROM transactions ORDER BY month').fetchall()
            
            self.income_data = []
            self.expense_data = []
            
            max_month = 0 if not data else max(row[0] for row in data)
            
            self.income_data = [None] * max_month
            self.expense_data = [None] * max_month
            
            for month, income, spending in data:
                month_idx = month - 1
                self.income_data[month_idx] = income
                self.expense_data[month_idx] = spending
                
        except Exception as e:
            messagebox.showerror("Database Error", f"Error loading data: {str(e)}")

    def setup_prediction_tab(self):
        for widget in self.predict_tab.winfo_children():
            widget.destroy()
            
        style = ttk.Style()
        style.configure("Custom.TEntry", padding=5, relief="flat")
        
        header_frame = tk.Frame(self.predict_tab, bg="white")
        header_frame.pack(pady=20)
        
        tk.Label(header_frame, 
                text="Smart Spending Predictor", 
                font=("Helvetica", 24, "bold"), 
                bg="white",
                fg="#2C3E50").pack()
                
        tk.Label(header_frame,
                text="Enter your last 5 months of financial data for AI-powered prediction",
                font=("Helvetica", 12),
                bg="white",
                fg="#7F8C8D").pack(pady=5)

        column_headers_frame = tk.Frame(self.predict_tab, bg="white")
        column_headers_frame.pack(pady=5)
        
        tk.Label(column_headers_frame,
                text="Month",
                font=("Helvetica", 12, "bold"),
                width=10,
                bg="white").pack(side=tk.LEFT, padx=5)
                
        tk.Label(column_headers_frame,
                text="Income ($)",
                font=("Helvetica", 12, "bold"),
                width=15,
                bg="white").pack(side=tk.LEFT, padx=5)
                
        tk.Label(column_headers_frame,
                text="Spending ($)",
                font=("Helvetica", 12, "bold"),
                width=15,
                bg="white").pack(side=tk.LEFT, padx=5)

        # Create a frame for entries with alternating colors
        entries_frame = tk.Frame(self.predict_tab, bg="white")
        entries_frame.pack(pady=20, padx=50)
        
        self.income_entries = []
        self.expense_entries = []
        
        for i in range(5):
            row_frame = tk.Frame(entries_frame, bg="#F8F9FA")
            row_frame.pack(fill="x", pady=2)
            
            month_label = tk.Label(row_frame, 
                                 text=f"Month {i+1}:", 
                                 font=("Helvetica", 11),
                                 width=10,
                                 bg="#F8F9FA")
            month_label.pack(side=tk.LEFT, padx=5)
            
            income_entry = ttk.Entry(row_frame, 
                                   style="Custom.TEntry",
                                   width=15)
            income_entry.pack(side=tk.LEFT, padx=5)
            self.income_entries.append(income_entry)
            
            tk.Label(row_frame, 
                    text="→", 
                    font=("Helvetica", 11),
                    bg="#F8F9FA").pack(side=tk.LEFT, padx=5)
                    
            expense_entry = ttk.Entry(row_frame,
                                    style="Custom.TEntry",
                                    width=15)
            expense_entry.pack(side=tk.LEFT, padx=5)
            self.expense_entries.append(expense_entry)

        predict_button = tk.Button(self.predict_tab,
                                 text="Generate AI Prediction",
                                 font=("Helvetica", 12, "bold"),
                                 bg="#2ECC71",
                                 fg="white",
                                 relief="flat",
                                 command=self.predict_next_month_spending)
        predict_button.pack(pady=20)
        
        self.results_frame = tk.Frame(self.predict_tab, bg="white")
        self.results_frame.pack(pady=20, fill="x", padx=50)
        
        self.prediction_label = tk.Label(self.results_frame,
                                       text="",
                                       font=("Helvetica", 14),
                                       bg="white")
        self.prediction_label.pack()

    def setup_visualization_tab(self):
        """Setup the visualization tab with its widgets"""
        for widget in self.visualize_tab.winfo_children():
            widget.destroy()

        header = tk.Label(self.visualize_tab, 
                         text="Income and Spending Visualization", 
                         font=("Helvetica", 16, "bold"),
                         bg="white")
        header.pack(pady=10)

        self.plot_frame = tk.Frame(self.visualize_tab, bg="white")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        update_button = tk.Button(self.visualize_tab,
                                text="Update Visualization",
                                font=("Helvetica", 12),
                                bg="#2ECC71",
                                fg="white",
                                command=self.visualize_data)
        update_button.pack(pady=10)

    def track_expenses(self):
        try:
            expenses = float(simpledialog.askstring("Track Your Expenses", "Enter your monthly expenses: $"))
            if expenses <= 5000:
                message = "Way to go! Your spending is within budget."
            else:
                message = "You're spending quite a bit. Consider reviewing your budget."
            messagebox.showinfo("Expenses", message)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid numeric value for expenses.")

    def see_money_tips(self):
        try:
            income = float(simpledialog.askstring("Income", "Enter your monthly income: $"))
            tips = ("1. Save at least 20% of your income.\n"
                    "2. Limit your fixed expenses to 50%.\n"
                    "3. Invest 30% of your income for growth.")
            messagebox.showinfo("Money Tips", tips)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid numeric value for income.")

    def train_model(self):
        """Train the model with the current data, filtering out None values"""
        try:
            valid_data = [(inc, exp) for inc, exp in zip(self.income_data, self.expense_data) 
                         if inc is not None and exp is not None and inc != 0 and exp != 0]
            
            if len(valid_data) >= 5:
                incomes, expenses = zip(*valid_data)
                
                X = np.array(incomes).reshape(-1, 1)
                y = np.array(expenses)
                
                self.scaler = StandardScaler().fit(X)
                X_scaled = self.scaler.transform(X)
                
                linear_model = LinearRegression()
                linear_model.fit(X_scaled, y)
                self.rf_model.fit(X_scaled, y)
                
                lr_score = linear_model.score(X_scaled, y)
                rf_score = self.rf_model.score(X_scaled, y)
                
                self.model = self.rf_model if rf_score > lr_score else linear_model
                return max(lr_score, rf_score)
                
            return 0
            
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            return 0

    def predict_next_month_spending(self):
        try:
            for income_entry, expense_entry in zip(self.income_entries, self.expense_entries):
                if not income_entry.get().strip() or not expense_entry.get().strip():
                    messagebox.showerror("Error", "Please fill all fields with values.")
                    return

            incomes = []
            expenses = []
            for income_entry, expense_entry in zip(self.income_entries, self.expense_entries):
                try:
                    income = float(income_entry.get().strip())
                    expense = float(expense_entry.get().strip())
                    if income < 0 or expense < 0:
                        messagebox.showerror("Error", "Values cannot be negative.")
                        return
                    incomes.append(income)  
                    expenses.append(expense)  
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numeric values (e.g., 1000.50)")
                    return

            if len(incomes) == 5 and len(expenses) == 5:
                for i, (income, expense) in enumerate(zip(incomes, expenses)):
                    month_num = i + 1  # Months are now in correct order
                    try:
                        self.cursor.execute('''
                            INSERT OR REPLACE INTO transactions (month, income, spending)
                            VALUES (?, ?, ?)
                        ''', (month_num, income, expense))
                    except Exception as e:
                        messagebox.showerror("Database Error", f"Error saving prediction data: {str(e)}")
                        return
                
                self.conn.commit()
                
                self.income_data = incomes
                self.expense_data = expenses
                
                spending_ratios = [e/i if i != 0 else 0 for e, i in zip(expenses, incomes)]
                
                X = np.array(range(5)).reshape(-1, 1)
                y_expense = np.array(expenses)
                y_ratio = np.array(spending_ratios)
                
                expense_reg = LinearRegression().fit(X, y_expense)
                ratio_reg = LinearRegression().fit(X, y_ratio)
                
                expense_trend = expense_reg.coef_[0]
                ratio_trend = ratio_reg.coef_[0]
                
                next_month_absolute = expense_reg.predict([[5]])[0]
                next_month_ratio = ratio_reg.predict([[5]])[0]
                
                income_variation = np.std(np.diff(incomes)) / np.mean(incomes) if np.mean(incomes) != 0 else 0
                
                if income_variation < 0.1:  
                    final_prediction = next_month_absolute
                    prediction_weight = "absolute"
                else:
                    income_reg = LinearRegression().fit(X, np.array(incomes))
                    next_month_income = income_reg.predict([[5]])[0]
                    final_prediction = next_month_ratio * next_month_income
                    prediction_weight = "ratio"
                
                final_prediction = max(0, final_prediction)
                
                expense_r2 = expense_reg.score(X, y_expense)
                ratio_r2 = ratio_reg.score(X, y_ratio)
                
                expense_consistency = 1 - (np.std(np.diff(expenses)) / (max(expenses) - min(expenses)) if max(expenses) != min(expenses) else 0)
                ratio_consistency = 1 - (np.std(np.diff(spending_ratios)) / (max(spending_ratios) - min(spending_ratios)) if max(spending_ratios) != min(spending_ratios) else 0)
                
                if prediction_weight == "absolute":
                    confidence = (expense_r2 * 0.7 + expense_consistency * 0.3) * 100
                else:
                    confidence = (ratio_r2 * 0.7 + ratio_consistency * 0.3) * 100
                
                self.update_prediction_display(final_prediction, confidence)
                self.add_enhanced_trend_analysis(expenses, incomes, spending_ratios, 
                                              expense_trend, ratio_trend, prediction_weight)
                
                if hasattr(self, 'update_transaction_list'):
                    self.update_transaction_list()

            else:
                messagebox.showerror("Error", "Please fill out all fields correctly.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def add_enhanced_trend_analysis(self, expenses, incomes, spending_ratios, 
                                  expense_trend, ratio_trend, prediction_weight):
        """Add detailed trend analysis including income relationships"""
        trend_frame = tk.Frame(self.results_frame, bg="white")
        trend_frame.pack(pady=10)
        
        expense_direction = "decreasing" if expense_trend < 0 else "increasing"
        ratio_direction = "decreasing" if ratio_trend < 0 else "increasing"
        
        trend_text = f"Absolute spending is {expense_direction} by ${abs(expense_trend):.2f} per month"
        ratio_text = f"Spending-to-income ratio is {ratio_direction} by {abs(ratio_trend)*100:.1f}% per month"
        method_text = f"Prediction based on: {'absolute spending' if prediction_weight == 'absolute' else 'income ratio'} trends"
        
        # Display average spending ratio
        avg_ratio = np.mean(spending_ratios) * 100
        ratio_text_avg = f"Average spending is {avg_ratio:.1f}% of income"
        
        # Display analysis
        for text in [trend_text, ratio_text, ratio_text_avg, method_text]:
            tk.Label(trend_frame,
                    text=text,
                    font=("Helvetica", 11),
                    bg="white",
                    fg="#34495E").pack()
        
        monthly_details = "Monthly spending/income ratios: " + ", ".join([f"{r*100:.1f}%" for r in spending_ratios])
        tk.Label(trend_frame,
                text=monthly_details,
                font=("Helvetica", 11),
                bg="white",
                fg="#34495E").pack()

    def update_prediction_display(self, prediction, confidence):

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        result_text = f"Predicted Spending: ${prediction:,.2f}"
        confidence_text = f"Confidence Level: {confidence:.1f}%"
        
        if len(self.expense_data) >= 2:
            recent_trend = self.expense_data[-1] - self.expense_data[-2]
            trend_text = "Spending is trending "
            trend_text += f"{'up' if recent_trend > 0 else 'down'} by ${abs(recent_trend):,.2f}"
        else:
            trend_text = "Not enough data to determine trend"
        
        tk.Label(self.results_frame,
                text=result_text,
                font=("Helvetica", 16, "bold"),
                bg="white",
                fg="#2C3E50").pack(pady=5)
                
        tk.Label(self.results_frame,
                text=confidence_text,
                font=("Helvetica", 12),
                bg="white",
                fg="#7F8C8D").pack(pady=5)
                
        tk.Label(self.results_frame,
                text=trend_text,
                font=("Helvetica", 10, "italic"),
                bg="white",
                fg="#34495E").pack(pady=5)

    def visualize_data(self):
        """Update the visualization with current data"""
        try:
            # Filter out None values for visualization
            valid_data = [(i+1, inc, exp) for i, (inc, exp) in enumerate(zip(self.income_data, self.expense_data)) 
                         if inc is not None and exp is not None]
            
            if valid_data:
                self.ax.clear()
                
                months, incomes, expenses = zip(*valid_data)

                self.ax.plot(months, incomes, 'b-o', label='Income', linewidth=2)
                self.ax.plot(months, expenses, 'r-o', label='Spending', linewidth=2)

                if len(months) >= 2:
                    z_income = np.polyfit(months, incomes, 1)
                    z_expense = np.polyfit(months, expenses, 1)
                    p_income = np.poly1d(z_income)
                    p_expense = np.poly1d(z_expense)
                    
                    self.ax.plot(months, p_income(months), 'b--', alpha=0.5)
                    self.ax.plot(months, p_expense(months), 'r--', alpha=0.5)

                self.ax.set_title('Income vs Spending Trends', pad=20, fontsize=14)
                self.ax.set_xlabel('Month', fontsize=12)
                self.ax.set_ylabel('Amount ($)', fontsize=12)
                self.ax.grid(True, linestyle='--', alpha=0.7)
                self.ax.legend(fontsize=10)
                
                self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                if self.canvas is None:
                    self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
                    toolbar.update()
                else:
                    self.canvas.draw()
                
            else:
                messagebox.showinfo("Info", "No data to visualize. Please enter at least one transaction.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the visualization: {str(e)}")

    def setup_input_tab(self):
        """Setup the transaction input tab"""
        for widget in self.input_tab.winfo_children():
            widget.destroy()

        tk.Label(self.input_tab, 
                text="Add Your Transactions", 
                font=("Helvetica", 16, "bold"), 
                bg="white").pack(pady=20)

        tk.Label(self.input_tab, 
                text="Month:", 
                font=("Helvetica", 12), 
                bg="white").pack(pady=5)
        self.new_month_entry = tk.Entry(self.input_tab, font=("Helvetica", 12))
        self.new_month_entry.pack(pady=5)

        tk.Label(self.input_tab, 
                text="Income ($):", 
                font=("Helvetica", 12), 
                bg="white").pack(pady=5)
        self.new_income_entry = tk.Entry(self.input_tab, font=("Helvetica", 12))
        self.new_income_entry.pack(pady=5)

        tk.Label(self.input_tab, 
                text="Spending ($):", 
                font=("Helvetica", 12), 
                bg="white").pack(pady=5)
        self.new_spending_entry = tk.Entry(self.input_tab, font=("Helvetica", 12))
        self.new_spending_entry.pack(pady=5)

        tk.Button(self.input_tab, 
                 text="Add Transaction",
                 font=("Helvetica", 12),
                 bg="#2ECC71",
                 fg="white",
                 command=self.add_transaction).pack(pady=10)

    def add_transaction(self):
        try:
            month = self.new_month_entry.get().strip()
            
            if month:
                try:
                    month_num = int(month)
                    if month_num < 1:
                        messagebox.showerror("Error", "Month number must be positive.")
                        return
                    month_index = month_num - 1
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid month number.")
                    return
            else:
                month_num = len(self.income_data) + 1
                month_index = len(self.income_data)

            new_income = float(self.new_income_entry.get())
            new_spending = float(self.new_spending_entry.get())

            if new_income < 0 or new_spending < 0:
                messagebox.showerror("Error", "Values cannot be negative.")
                return

            try:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO transactions (month, income, spending)
                    VALUES (?, ?, ?)
                ''', (month_num, new_income, new_spending))
                self.conn.commit()
            except Exception as e:
                messagebox.showerror("Database Error", f"Error saving transaction: {str(e)}")
                return

            if month_index >= len(self.income_data):
                self.income_data.extend([None] * (month_index - len(self.income_data) + 1))
                self.expense_data.extend([None] * (month_index - len(self.expense_data) + 1))
            
            self.income_data[month_index] = new_income
            self.expense_data[month_index] = new_spending

            if len(self.income_data) >= 5:
                self.train_model()

            messagebox.showinfo("Success", f"Transaction for Month {month_num} {'updated' if month_index < len(self.income_data) else 'added'} successfully!")

            self.new_month_entry.delete(0, tk.END)
            self.new_income_entry.delete(0, tk.END)
            self.new_spending_entry.delete(0, tk.END)

            if hasattr(self, 'update_transaction_list'):
                self.update_transaction_list()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for income and spending.")

    def setup_manage_transactions_tab(self):
        """Setup the manage transactions tab"""
        for widget in self.manage_tab.winfo_children():
            widget.destroy()

        main_container = tk.Frame(self.manage_tab, bg="white")
        main_container.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_container, 
                text="Manage Your Transactions", 
                font=("Helvetica", 20, "bold"),
                bg="white").pack(pady=20)

        self.transactions_frame = tk.Frame(main_container, bg="white")
        self.transactions_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        self.transactions_frame.grid_columnconfigure(0, weight=1)  
        self.transactions_frame.grid_columnconfigure(1, weight=1)  
        self.transactions_frame.grid_columnconfigure(2, weight=1)  
        self.transactions_frame.grid_columnconfigure(3, weight=1)  

        tk.Label(self.transactions_frame, text="Month", font=("Helvetica", 12, "bold"), 
                bg="white").grid(row=0, column=0, pady=10, sticky="n")
        tk.Label(self.transactions_frame, text="Income ($)", font=("Helvetica", 12, "bold"), 
                bg="white").grid(row=0, column=1, pady=10)
        tk.Label(self.transactions_frame, text="Spending ($)", font=("Helvetica", 12, "bold"), 
                bg="white").grid(row=0, column=2, pady=10)
        tk.Label(self.transactions_frame, text="Actions", font=("Helvetica", 12, "bold"), 
                bg="white").grid(row=0, column=3, pady=10)

        self.update_transaction_list()

    def update_transaction_list(self):
        """Update the list of transactions"""
        for widget in self.transactions_frame.winfo_children():
            if int(widget.grid_info()['row']) > 0:  # Keep header row
                widget.destroy()

        if self.income_data and self.expense_data:
            for i, (income, expense) in enumerate(zip(self.income_data, self.expense_data)):
                if income is None and expense is None:
                    continue

                row = i + 1                
                
                if income == 0 and expense == 0:
                    month_text = f"Month {i+1} (Empty)"
                    income_text = "---"
                    expense_text = "---"
                else:
                    month_text = f"Month {i+1}"
                    income_text = f"${income:,.2f}"
                    expense_text = f"${expense:,.2f}"

                tk.Label(self.transactions_frame,
                        text=month_text,
                        font=("Helvetica", 11),
                        bg="white").grid(row=row, column=0, pady=5)

                tk.Label(self.transactions_frame,
                        text=income_text,
                        font=("Helvetica", 11),
                        bg="white").grid(row=row, column=1, pady=5)

                tk.Label(self.transactions_frame,
                        text=expense_text,
                        font=("Helvetica", 11),
                        bg="white").grid(row=row, column=2, pady=5)

                tk.Button(self.transactions_frame,
                         text="Delete" if income != 0 or expense != 0 else "Remove",
                         font=("Helvetica", 10),
                         bg="#E74C3C",
                         fg="white",
                         command=lambda x=i: self.confirm_delete_transaction(x)).grid(row=row, column=3, pady=5)

        else:
            tk.Label(self.transactions_frame,
                    text="No transactions to display",
                    font=("Helvetica", 11, "italic"),
                    bg="white").grid(row=1, column=0, columnspan=4, pady=20)

    def confirm_delete_transaction(self, index):
        """Show confirmation dialog before deleting a transaction"""
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete the transaction for Month {index+1}?"
        )
        
        if result:
            self.delete_transaction(index)

    def delete_transaction(self, index):
        """Delete a transaction and update the display"""
        try:
            month_num = index + 1
            
            if self.income_data[index] == 0 and self.expense_data[index] == 0:
                self.cursor.execute('DELETE FROM transactions WHERE month = ?', (month_num,))
                self.income_data[index] = None
                self.expense_data[index] = None
                message = f"Transaction for Month {month_num} permanently removed from visualization!"
            else:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO transactions (month, income, spending)
                    VALUES (?, 0, 0)
                ''', (month_num,))
                self.income_data[index] = 0
                self.expense_data[index] = 0
                message = f"Transaction for Month {month_num} deleted successfully!"
            
            self.conn.commit()
            
            self.update_transaction_list()
            
            messagebox.showinfo("Success", message)
            
            if len(self.income_data) >= 5:
                self.train_model()
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while deleting the transaction: {str(e)}")

    def __del__(self):
        """Destructor to properly close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalFinanceApp(root)
    root.mainloop()
