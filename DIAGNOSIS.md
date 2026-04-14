# Ground Truth Debug Checklist

## Step 1: Verify Ground Truth File Format
Your ground truth file (CSV or JSON) must have these exact columns:

### CSV Format (Recommended)
```
question,expected_answer,context
"What is the employee benefits at YASH?","YASH offers health insurance and retirement plans","See Employee Handbook Section 4"
"How many leave days do YASH employees get?","10 days per year, extendable to 15 for managers","Leave Policy Document v2"
```

### JSON Format (Alternative)
```json
[
  {
    "question": "What is the employee benefits at YASH?",
    "expected_answer": "YASH offers health insurance and retirement plans",
    "context": "See Employee Handbook Section 4"
  },
  {
    "question": "How many leave days do YASH employees get?", 
    "expected_answer": "10 days per year, extendable to 15 for managers",
    "context": "Leave Policy Document v2"
  }
]
```

## Step 2: Verify Upload in Configure Page
1. Go to Configure Step 1
2. Check **"Ground Truth File — question, expected_answer & context (CSV or JSON)"** section
3. Upload your file
4. Verify the filename appears in the field

## Step 3: Verify File Size
- **Minimum**: 3 rows (headers + 2 data rows)
- **RAG Knowledge Base**: First 800 chars ONLY
- **Solution**: If your YASH context is >800 chars, either:
  - Add it to the `context` column, OR
  - Request backend modification to use full file for RAG

## Step 4: Expected Behavior After Fix
If ground truth is properly loaded:
1. navigate to Preview page
2. Click **Run Test Suite**
3. Results should show prompts that are **EXACT questions from your CSV**
4. NOT LLM-generated HR questions

## Current Issues Identified
1. ✅ Async FileReader race condition = FIXED
2. ✅ Feedback loop missing ground_truth = FIXED
3. ✅ Azure endpoint normalization = FIXED
4. ❌ **RAG Knowledge Base limited to 800 chars** = NEEDS FIX
5. ❌ Ground truth file may not be uploading/parsing = TEST REQUIRED

