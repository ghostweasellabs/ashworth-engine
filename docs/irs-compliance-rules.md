# IRS Compliance Rules for Ashworth Engine

## Overview
Official IRS compliance rules and guidelines that should be stored in the RAG system for dynamic retrieval by agents processing tax-related information.

## Core Principles

1. **Zero Hallucination Policy**
   - All tax-related processing must be grounded in verifiable tax rules and documentation
   - Cross-reference all tax advice with official IRS publications
   - Never generate tax advice not explicitly supported by official documentation

2. **Authoritative Source Requirement**
   - Use current IRS Publication 334 (Tax Guide for Small Business) as the primary reference
   - Verify publication currency before implementing tax-related features
   - Discontinue use of outdated publications (e.g., Publication 535)

3. **Conservative, Audit-Defensible Approach**
   - Use conservative interpretations of tax rules
   - When in doubt, recommend consultation with a tax professional
   - Document all assumptions and interpretations

## Specific Tax Rules

### Expense Categorization Standards

1. **11 Official IRS Expense Categories**
   - Advertising and promotion
   - Bad debts
   - Car and truck expenses
   - Commissions and fees
   - Contract labor
   - Depletion
   - Depreciation
   - Employee benefit programs
   - Insurance (other than health)
   - Interest (mortgage)
   - Legal and professional services
   - Office expense
   - Pension and profit-sharing plans
   - Rent or lease of vehicles, machinery, equipment
   - Rent or lease of property (other than vehicles)
   - Repairs and maintenance
   - Supplies (office)
   - Taxes and licenses
   - Travel (business meals, lodging, transportation)
   - Utilities

2. **"Ordinary and Necessary" Standard**
   - Ordinary: Common and accepted in your industry
   - Necessary: Helpful and appropriate for your business
   - Both conditions must be met for deduction eligibility

### Deduction Rules and Limitations

1. **Business Meal Deduction**
   - Only 50% of business meal costs are deductible
   - Must be associated with business discussion
   - Must be with clients, customers, or employees

2. **Section 179 Deduction**
   - Maximum deduction: $1,080,000 for 2023 (subject to phase-out)
   - Total equipment purchases limited to $2,700,000 for 2023
   - Property must be placed in service during the tax year
   - Cannot create/ increase a net loss

3. **Vehicle Expense Deduction**
   - Standard mileage rate or actual expenses
   - Personal use portion not deductible
   - Keep detailed records of business use

### Reporting Requirements

1. **Form 8300 Reporting**
   - Report cash payments over $10,000
   - File within 15 days of receipt
   - Include payer and recipient information

2. **Record Keeping Requirements**
   - Keep records for at least 3 years
   - Document business purpose for all expenses
   - Maintain receipts and supporting documentation

## Implementation Guidance

### For Tax Categorizer Agent

1. Always reference the 11 official IRS expense categories
2. Apply the "Ordinary and Necessary" standard to all categorizations
3. Flag potential audit triggers for review
4. Provide conservative interpretations of ambiguous cases

### For Report Generator Agent

1. Clearly cite IRS publications when providing tax advice
2. Include disclaimers about the need for professional consultation
3. Present multiple interpretations when rules are ambiguous
4. Document all assumptions made in calculations

## Audit and Compliance

### Audit Trigger Identification

1. **High-Risk Expense Categories**
   - Travel and entertainment expenses
   - Vehicle expenses without detailed logs
   - Home office deductions
   - Meals and entertainment without business context

2. **Documentation Deficiencies**
   - Missing receipts
   - Inadequate business purpose documentation
   - Insufficient record keeping

### Compliance Verification

1. **Regular Updates**
   - Monitor for IRS publication updates
   - Review and update rules annually
   - Stay informed about tax law changes

2. **Quality Assurance**
   - Cross-reference with multiple IRS sources
   - Validate interpretations with tax professionals
   - Test edge cases and ambiguous scenarios

## Examples

### Correct Application

When categorizing a business lunch with a client:
- Category: Travel (business meals, lodging, transportation)
- Deduction: 50% of cost
- Documentation: Receipt and notes about business discussion

### Incorrect Application

When categorizing a personal meal:
- Not deductible as business expense
- Should not be included in business expense reports
- Explanation: Does not meet "Ordinary and Necessary" standard

## References

- IRS Publication 334 (Tax Guide for Small Business)
- IRS Publication 587 (Business Use of Your Home)
- IRS Publication 463 (Travel, Gift, and Car Expenses)
- Form 8300 instructions