ctc = float(input("Enter your Total CTC (₹): "))
bonus = float(input("Enter your Bonus Amount (₹): "))
deduction = float(input("Enter total deductions (₹): "))

total_income = ctc + bonus
taxable_income = max(total_income - deduction, 0)

taxable_above_250000 = max(taxable_income - 250000, 0)
tax = taxable_above_250000 * 0.05

print(f"Total Income (CTC + Bonus): ₹{total_income}")
print(f"Total deductions: ₹{deduction}")
print(f"Taxable Income: ₹{taxable_income}")
print(f"Tax Payable: ₹{tax:.2f}")
