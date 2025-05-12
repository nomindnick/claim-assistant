#!/usr/bin/env python
"""
Create a test dataset with known document boundaries for evaluating segmentation.

This script generates synthetic multi-document PDFs with clear boundaries for testing
document segmentation algorithms. It creates a variety of document types and combinations
to challenge boundary detection algorithms with different scenarios.

Usage:
    python -m tests.create_test_dataset [--output-dir PATH] [--count NUM]
"""

import os
import argparse
import random
import json
from pathlib import Path

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Install reportlab if needed
try:
    import reportlab
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "reportlab"])

# Document templates for different document types
DOCUMENT_TEMPLATES = {
    "meeting_minutes": """
{header}
DATE: {date}
PROJECT: {project}
LOCATION: {location}

ATTENDEES:
{attendees}

AGENDA ITEMS:
{agenda_items}

ACTION ITEMS:
{action_items}

NEXT MEETING: {next_meeting}
""",
    "change_order": """
{header}
CHANGE ORDER #{number}
DATE: {date}
PROJECT: {project}

DESCRIPTION OF CHANGE:
{description}

REASON FOR CHANGE:
{reason}

COST IMPACT: ${cost}
SCHEDULE IMPACT: {schedule} days

APPROVED BY:
{approved_by}
""",
    "daily_report": """
{header}
DAILY REPORT
DATE: {date}
PROJECT: {project}
WEATHER: {weather}

WORK PERFORMED:
{work_performed}

LABOR:
{labor}

EQUIPMENT:
{equipment}

ISSUES/DELAYS:
{issues}
""",
    "email": """
{header}
From: {sender}
To: {recipient}
Subject: {subject}
Date: {date}

{body}

{signature}
""",
    "invoice": """
{header}
INVOICE #{number}
DATE: {date}
DUE DATE: {due_date}

BILL TO:
{bill_to}

DESCRIPTION OF SERVICES:
{description}

SUBTOTAL: ${subtotal}
TAX: ${tax}
TOTAL: ${total}

PAYMENT TERMS: {payment_terms}
""",
    "letter": """
{header}
{sender_address}

{date}

{recipient_address}

Dear {recipient_name},

{body}

Sincerely,
{sender_name}
{sender_title}
""",
    "submittal": """
{header}
SUBMITTAL
SUBMITTAL NO: {number}
DATE: {date}
PROJECT: {project}

SPECIFICATION SECTION: {spec_section}
DESCRIPTION: {description}

SUBMITTED BY:
{submitted_by}

STATUS: {status}
""",
    "rfi": """
{header}
REQUEST FOR INFORMATION
RFI #: {number}
DATE: {date}
PROJECT: {project}

QUESTION:
{question}

RESPONSE:
{response}

IMPACT: {impact}
""",
}

# Sample data for generating document content
SAMPLE_DATA = {
    "projects": [
        "City Center Construction", 
        "Harbor Bridge Rehabilitation",
        "Mountain View Apartments", 
        "Westside Hospital Expansion",
        "Downtown Office Tower", 
        "Municipal Airport Renovation"
    ],
    "companies": [
        "Smith Construction Co.", 
        "Johnson Builders, Inc.",
        "Alpha Contractors", 
        "Metro Development Group",
        "Reliable Engineering", 
        "Pinnacle Construction"
    ],
    "locations": [
        "Main Conference Room", 
        "Site Office", 
        "Project Trailer",
        "Virtual Meeting", 
        "Client Office", 
        "Engineering Department"
    ],
    "people": [
        "John Smith (Project Manager)",
        "Mary Johnson (Architect)",
        "Robert Williams (Engineer)",
        "Sarah Brown (Owner Rep)",
        "Michael Davis (Superintendent)",
        "Jennifer Wilson (Quality Control)",
        "David Thompson (Subcontractor)",
        "Elizabeth Martinez (Inspector)",
        "James Anderson (Consultant)"
    ],
    "weather": [
        "Sunny, 75°F", 
        "Partly Cloudy, 68°F", 
        "Rain, 62°F",
        "Overcast, 70°F", 
        "Clear, 80°F", 
        "Windy, 65°F"
    ],
    "issues": [
        "Material delivery delayed by 2 days",
        "Concrete subcontractor short-staffed",
        "Rain interrupted exterior work",
        "Unmarked utility discovered during excavation",
        "Inspector required additional testing",
        "Equipment breakdown caused delay",
        "None reported"
    ],
    "work_performed": [
        "Excavation in Area A",
        "Concrete pour for footings",
        "Structural steel erection",
        "MEP rough-in on 2nd floor",
        "Drywall installation",
        "Site grading and drainage",
        "Painting in Area B",
        "Exterior brick veneer",
        "HVAC equipment installation"
    ],
    "change_reasons": [
        "Unforeseen site condition",
        "Owner-requested design change",
        "Code compliance requirement",
        "Coordination with existing utilities",
        "Material substitution due to availability",
        "Additional scope requested by client",
        "Design error or omission",
        "Weather-related mitigation measures"
    ],
    "email_subjects": [
        "Schedule Update for {project}",
        "Upcoming Inspection for {project}",
        "Revised Drawings for {project}",
        "Budget Concerns on {project}",
        "Subcontractor Coordination for {project}",
        "Material Delivery Schedule for {project}",
        "RFI Response: {project}",
        "Owner Meeting Summary: {project}"
    ]
}

def random_date(start_year=2023, end_year=2025):
    """Generate a random date string."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{month}/{day}/{year}"

def random_money(min_amount=1000, max_amount=100000):
    """Generate a random dollar amount."""
    return "{:,.2f}".format(random.randint(min_amount, max_amount))

def generate_document_content(doc_type):
    """Generate content for a specific document type."""
    project = random.choice(SAMPLE_DATA["projects"])
    company = random.choice(SAMPLE_DATA["companies"])
    
    # Common data for all document types
    common_data = {
        "header": f"{company.upper()}",
        "date": random_date(),
        "project": project,
    }
    
    if doc_type == "meeting_minutes":
        attendees = "\n".join([f"- {person}" for person in 
                              random.sample(SAMPLE_DATA["people"], k=random.randint(3, 6))])
        agenda_items = "\n".join([f"{i+1}. {item}" for i, item in 
                                 enumerate(random.sample(SAMPLE_DATA["work_performed"], k=random.randint(3, 5)))])
        action_items = "\n".join([f"- {person.split(' ')[0]}: {action}" for person, action in 
                                 zip(random.sample(SAMPLE_DATA["people"], k=random.randint(2, 4)),
                                     random.sample(SAMPLE_DATA["work_performed"], k=random.randint(2, 4)))])
        
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            location=random.choice(SAMPLE_DATA["locations"]),
            attendees=attendees,
            agenda_items=agenda_items,
            action_items=action_items,
            next_meeting=random_date()
        )
    
    elif doc_type == "change_order":
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            number=random.randint(1, 50),
            description=random.choice(SAMPLE_DATA["work_performed"]),
            reason=random.choice(SAMPLE_DATA["change_reasons"]),
            cost=random_money(),
            schedule=random.randint(-5, 20),
            approved_by=random.choice(SAMPLE_DATA["people"])
        )
    
    elif doc_type == "daily_report":
        work_performed = "\n".join([f"- {item}" for item in 
                                   random.sample(SAMPLE_DATA["work_performed"], k=random.randint(3, 6))])
        labor = "\n".join([f"- {random.randint(2, 10)} {trade}" for trade in 
                          ["Laborers", "Carpenters", "Electricians", "Plumbers", "Iron Workers", "Operators"]])
        equipment = "\n".join([f"- {equipment}" for equipment in 
                              ["Excavator", "Crane", "Bulldozer", "Backhoe", "Forklift", "Skid Steer"]])
        
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            weather=random.choice(SAMPLE_DATA["weather"]),
            work_performed=work_performed,
            labor=labor,
            equipment=equipment,
            issues=random.choice(SAMPLE_DATA["issues"])
        )
    
    elif doc_type == "email":
        sender = random.choice(SAMPLE_DATA["people"])
        recipient = random.choice(SAMPLE_DATA["people"])
        while recipient == sender:
            recipient = random.choice(SAMPLE_DATA["people"])
            
        subject = random.choice(SAMPLE_DATA["email_subjects"]).format(project=project)
        body_paragraphs = []
        for _ in range(random.randint(2, 4)):
            # Create plausible email paragraphs
            topic = random.choice(["schedule", "budget", "quality", "safety", "design"])
            if topic == "schedule":
                body_paragraphs.append(f"Regarding the project schedule, we are currently {random.choice(['ahead of', 'behind', 'on'])} schedule. The {random.choice(SAMPLE_DATA['work_performed']).lower()} {random.choice(['is scheduled for', 'was completed on', 'will begin on'])} {random_date()}.")
            elif topic == "budget":
                body_paragraphs.append(f"The budget for {random.choice(SAMPLE_DATA['work_performed']).lower()} is currently {random.choice(['over', 'under', 'within'])} the allocated amount by ${random.randint(1000, 50000)}. We may need to {random.choice(['reduce scope', 'request additional funds', 'reallocate from other line items'])}.")
            elif topic == "quality":
                body_paragraphs.append(f"Our quality control team has {random.choice(['identified an issue with', 'completed inspection of', 'approved'])} the {random.choice(SAMPLE_DATA['work_performed']).lower()}. {random.choice(['Remedial work is needed.', 'Everything meets specifications.', 'Please review the attached report.'])}")
            elif topic == "safety":
                body_paragraphs.append(f"There was a {random.choice(['safety incident', 'near miss', 'safety audit'])} related to {random.choice(SAMPLE_DATA['work_performed']).lower()}. {random.choice(['No injuries were reported.', 'One worker required first aid.', 'Additional safety measures have been implemented.'])}")
            else:  # design
                body_paragraphs.append(f"The design team has {random.choice(['issued revised drawings for', 'requested clarification on', 'approved the submittals for'])} {random.choice(SAMPLE_DATA['work_performed']).lower()}. {random.choice(['Please review at your earliest convenience.', 'This resolves RFI #123.', 'Implementation will begin next week.'])}")
                
        body = "\n\n".join(body_paragraphs)
        signature = f"\n{sender}\n{random.choice(SAMPLE_DATA['companies'])}\n{random.choice(['555-123-4567', '555-987-6543'])}"
        
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            sender=sender,
            recipient=recipient,
            subject=subject,
            body=body,
            signature=signature
        )
    
    elif doc_type == "invoice":
        subtotal = float(random_money(min_amount=5000, max_amount=200000).replace(",", ""))
        tax_rate = random.uniform(0.05, 0.09)
        tax = subtotal * tax_rate
        total = subtotal + tax
        
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            number=random.randint(1000, 9999),
            due_date=random_date(),
            bill_to=f"{random.choice(SAMPLE_DATA['companies'])}\nAttn: {random.choice(SAMPLE_DATA['people']).split(' ')[0]}\n123 Business St.\nCity, State 12345",
            description="\n".join([f"- {item}: ${random_money(min_amount=1000, max_amount=50000)}" for item in 
                                 random.sample(SAMPLE_DATA["work_performed"], k=random.randint(3, 6))]),
            subtotal="{:,.2f}".format(subtotal),
            tax="{:,.2f}".format(tax),
            total="{:,.2f}".format(total),
            payment_terms=random.choice(["Net 30", "Net 15", "Due on Receipt"])
        )
    
    elif doc_type == "letter":
        sender_name = random.choice(SAMPLE_DATA["people"])
        sender_title = sender_name.split("(")[1].replace(")", "") if "(" in sender_name else "Manager"
        sender_name = sender_name.split(" (")[0]
        
        recipient_name = random.choice(SAMPLE_DATA["people"])
        recipient_title = recipient_name.split("(")[1].replace(")", "") if "(" in recipient_name else "Manager"
        recipient_name = recipient_name.split(" (")[0]
        
        body_paragraphs = []
        for _ in range(random.randint(2, 4)):
            # Create plausible letter paragraphs
            topic = random.choice(["formal notice", "update", "request", "appreciation"])
            if topic == "formal notice":
                body_paragraphs.append(f"This letter serves as formal notice regarding {random.choice(['changes to', 'delays in', 'completion of'])} {random.choice(SAMPLE_DATA['work_performed']).lower()} on the {project} project. As of {random_date()}, we have {random.choice(['encountered unforeseen conditions', 'received revised instructions', 'completed the specified work'])}.")
            elif topic == "update":
                body_paragraphs.append(f"I am writing to provide an update on the progress of {project}. The {random.choice(SAMPLE_DATA['work_performed']).lower()} is now {random.randint(50, 100)}% complete and {random.choice(['ahead of schedule', 'on schedule', 'slightly behind schedule'])}. We anticipate {random.choice(['completion by', 'additional challenges with', 'requiring assistance on'])} this portion of the work.")
            elif topic == "request":
                body_paragraphs.append(f"We respectfully request {random.choice(['additional information regarding', 'approval for changes to', 'an extension for'])} the {random.choice(SAMPLE_DATA['work_performed']).lower()}. This is necessary due to {random.choice(['unforeseen site conditions', 'material delivery delays', 'design modifications'])} that have impacted our work.")
            else:  # appreciation
                body_paragraphs.append(f"I would like to express our appreciation for your {random.choice(['prompt response to', 'assistance with', 'cooperation regarding'])} the recent {random.choice(['issue', 'change', 'milestone'])} on the {project} project. Your {random.choice(['team expertise', 'timely input', 'professional approach'])} has been instrumental in {random.choice(['resolving challenges', 'maintaining progress', 'achieving quality standards'])}.")
        
        body = "\n\n".join(body_paragraphs)
        
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            sender_address=f"{company}\n123 Business St.\nCity, State 12345",
            recipient_address=f"{random.choice(SAMPLE_DATA['companies'])}\nAttn: {recipient_name}\n456 Corporate Ave.\nCity, State 12345",
            recipient_name=recipient_name,
            body=body,
            sender_name=sender_name,
            sender_title=sender_title
        )
    
    elif doc_type == "submittal":
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            number=f"SUB-{random.randint(100, 999)}",
            spec_section=f"{random.randint(1, 33)} {random.randint(10, 99)} {random.randint(10, 99)}",
            description=random.choice([
                "Concrete Mix Design", 
                "Structural Steel Shop Drawings",
                "Mechanical Equipment Specifications",
                "Electrical Panel Schedule",
                "Window System Product Data",
                "Roofing Material Samples",
                "Fire Protection System Layout",
                "Site Drainage Plan"
            ]),
            submitted_by=f"{random.choice(SAMPLE_DATA['people'])}\n{random.choice(SAMPLE_DATA['companies'])}",
            status=random.choice(["Approved", "Approved as Noted", "Revise and Resubmit", "Rejected", "For Information Only"])
        )
    
    elif doc_type == "rfi":
        return DOCUMENT_TEMPLATES[doc_type].format(
            **common_data,
            number=random.randint(1, 100),
            question=f"Please clarify the {random.choice(['specifications', 'drawing details', 'dimensions', 'material requirements'])} for {random.choice(SAMPLE_DATA['work_performed']).lower()}. {random.choice(['The drawings show', 'The specifications indicate', 'There is a conflict between'])} {random.choice(['conflicting information', 'insufficient detail', 'unclear requirements'])}.",
            response=f"Based on {random.choice(['review of the documents', 'discussion with the design team', 'field conditions'])}, {random.choice(['proceed with', 'modify the installation to', 'refer to detail'])} {random.choice(['the higher standard', 'the latest revision', 'as discussed in the field'])}. {random.choice(['Additional details will be provided', 'No further action is required', 'A change order may be required'])}.",
            impact=random.choice(["No cost or schedule impact", f"Potential ${random_money()} cost impact", f"Potential {random.randint(1, 10)} day schedule impact", "To be determined"])
        )
    
    # Default fallback
    return f"DOCUMENT TYPE: {doc_type}\nDATE: {random_date()}\nPROJECT: {project}\n\nSample content for document type."

def create_document_pages(doc_type, page_count=1):
    """Generate a list of pages for a document of specified type and length."""
    content = generate_document_content(doc_type)
    
    # For multi-page documents, generate more content
    if page_count > 1:
        paragraphs = []
        # Add the main content first
        paragraphs.append(content)
        
        # Add additional pages of content
        for _ in range(page_count - 1):
            if doc_type == "meeting_minutes":
                topic = random.choice(["Progress Update", "Schedule Discussion", "Budget Review", "Quality Control", "Safety Issues"])
                paragraphs.append(f"\n\nDISCUSSION ITEM: {topic}\n\n" + 
                                  "\n".join([random.choice([
                                      "The team reviewed progress on recent activities.",
                                      "Several concerns were raised about the current timeline.",
                                      "Budget overruns were discussed for several line items.",
                                      "Quality issues on recent installations were addressed.",
                                      "Safety incident reports were reviewed.",
                                      "Subcontractor coordination issues were discussed.",
                                      "Material delivery schedules were updated.",
                                      "Design changes were presented by the architect.",
                                      "Permitting constraints were addressed.",
                                      "Weather impacts were assessed for the coming month."
                                  ]) for _ in range(random.randint(3, 6))]))
            
            elif doc_type in ["invoice", "change_order"]:
                paragraphs.append("\n\nDETAILED BREAKDOWN:\n\n" + 
                                 "\n".join([f"Item {i+1}: {random.choice(SAMPLE_DATA['work_performed'])} - ${random_money()}" 
                                          for i in range(random.randint(5, 10))]))
                
            elif doc_type == "email":
                paragraphs.append("\n\nADDITIONAL INFORMATION:\n\n" + 
                                 "\n".join([random.choice([
                                     "Please review the attached files for more details.",
                                     "The latest schedule shows several critical path activities.",
                                     "According to our records, this issue first appeared in January.",
                                     "The client has requested additional information about this matter.",
                                     "Our team is working diligently to resolve these concerns.",
                                     "Previous correspondence indicated that this was a priority.",
                                     "The subcontractors have been notified about these changes.",
                                     "We will need to coordinate this with other ongoing activities."
                                 ]) for _ in range(random.randint(3, 5))]))
                
            else:
                # Generic additional content for other document types
                paragraphs.append("\n\nADDITIONAL DETAILS:\n\n" + 
                                 "\n".join([random.choice([
                                     f"Section {random.randint(1, 10)}: {random.choice(SAMPLE_DATA['work_performed'])}",
                                     f"Status update as of {random_date()}: {random.randint(10, 100)}% complete",
                                     f"Issue identified: {random.choice(SAMPLE_DATA['issues'])}",
                                     f"Team involved: {', '.join(random.sample([name.split(' ')[0] for name in SAMPLE_DATA['people']], k=random.randint(2, 4)))}",
                                     f"Reference document: {random.choice(['Drawing A-', 'Specification ', 'Detail '])}{random.randint(1, 100)}",
                                     f"Location: {random.choice(['Building A', 'Building B', 'East Wing', 'West Wing', 'Phase 1', 'Phase 2'])} - {random.choice(['Level', 'Floor', 'Area'])}{random.randint(1, 5)}"
                                 ]) for _ in range(random.randint(5, 8))]))
    
        content = "\n".join(paragraphs)
    
    # Split into pages (approximate)
    content_length = len(content)
    chars_per_page = content_length // page_count
    
    pages = []
    for i in range(page_count):
        start = i * chars_per_page
        end = start + chars_per_page if i < page_count - 1 else content_length
        page_content = content[start:end]
        
        # Add page number for multi-page documents
        if page_count > 1:
            page_content += f"\n\nPage {i+1} of {page_count}"
            
        pages.append(page_content)
    
    return pages

def create_multi_document_pdf(output_path, document_specs):
    """
    Create a multi-document PDF with known boundaries.
    
    Args:
        output_path: Path where to save the PDF
        document_specs: List of (document_type, page_count) tuples
    
    Returns:
        Dictionary with document information and boundary positions
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    
    # Create a custom style for document headers
    header_style = ParagraphStyle(
        'Header',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12
    )
    
    # Prepare content
    story = []
    
    # Track document boundaries
    boundaries = []
    current_position = 0
    documents_info = []
    
    # Add each document with its pages
    for i, (doc_type, page_count) in enumerate(document_specs):
        doc_start_position = current_position
        
        # Generate document pages
        pages = create_document_pages(doc_type, page_count)
        
        for j, page_content in enumerate(pages):
            # Add page content
            if j == 0 and i > 0:
                # Add document separator (only between documents, not at the beginning)
                separator = Paragraph("=" * 50, normal_style)
                story.append(separator)
                story.append(Spacer(1, 0.25 * inch))
                current_position += len("=" * 50) + 1  # +1 for the spacer
                
                # Record boundary position
                boundaries.append({
                    'position': current_position,
                    'previous_doc': document_specs[i-1][0],
                    'next_doc': doc_type,
                    'confidence': 1.0  # Ground truth confidence is always 1.0
                })
            
            # Add the page content
            p = Paragraph(page_content, normal_style)
            story.append(p)
            story.append(Spacer(1, 0.5 * inch))
            current_position += len(page_content) + 1  # +1 for the spacer
    
        # Document info
        doc_end_position = current_position
        documents_info.append({
            'type': doc_type,
            'start_position': doc_start_position,
            'end_position': doc_end_position,
            'length': doc_end_position - doc_start_position,
            'page_count': page_count
        })
    
    # Build PDF
    doc.build(story)
    
    # Create metadata for the test dataset
    metadata = {
        'filename': os.path.basename(output_path),
        'document_count': len(document_specs),
        'document_types': [spec[0] for spec in document_specs],
        'total_pages': sum(spec[1] for spec in document_specs),
        'boundaries': boundaries,
        'documents': documents_info
    }
    
    return metadata

def generate_test_dataset(output_dir, count=5):
    """
    Generate a complete test dataset with multiple PDFs containing known document boundaries.
    
    Args:
        output_dir: Directory to save the PDFs and metadata
        count: Number of test PDFs to generate
    """
    # Create directories
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Document type combinations for testing
    document_combinations = [
        # Simple 2-document combinations
        [("email", 1), ("change_order", 1)],
        [("meeting_minutes", 2), ("daily_report", 1)],
        [("invoice", 1), ("letter", 1)],

        # More complex 3-document combinations
        [("email", 1), ("meeting_minutes", 2), ("invoice", 1)],
        [("rfi", 1), ("submittal", 1), ("change_order", 1)],

        # Challenging 4+ document combinations
        [("email", 1), ("meeting_minutes", 1), ("change_order", 1), ("daily_report", 1)],
        [("letter", 1), ("invoice", 2), ("email", 1), ("submittal", 1), ("rfi", 1)],

        # Very large mixed document
        [("meeting_minutes", 3), ("daily_report", 2), ("change_order", 1), ("invoice", 2),
         ("email", 2), ("rfi", 1), ("submittal", 1), ("letter", 1)],

        # Challenging scenarios with similar content
        [("email", 1), ("email", 1), ("email", 1)],  # Multiple emails with similar topics

        # Documents with shared terminology
        [("change_order", 1), ("rfi", 1), ("change_order", 1)],

        # Very short documents
        [("email", 1), ("submittal", 1), ("email", 1), ("submittal", 1)],

        # Documents with minimal formatting differences
        [("daily_report", 1), ("daily_report", 1), ("daily_report", 1)],

        # Mix of short and long documents
        [("email", 1), ("meeting_minutes", 3), ("email", 1), ("invoice", 1)]
    ]
    
    # Generate additional random combinations
    for _ in range(max(0, count - len(document_combinations))):
        num_docs = random.randint(2, 6)
        doc_types = random.sample(list(DOCUMENT_TEMPLATES.keys()), k=num_docs)
        random_combo = [(doc_type, random.randint(1, 3)) for doc_type in doc_types]
        document_combinations.append(random_combo)
    
    # Select the requested number of combinations
    selected_combinations = document_combinations[:count]
    
    # Generate each test PDF
    dataset_metadata = []
    for i, doc_combo in enumerate(selected_combinations):
        pdf_name = f"test_multi_doc_{i+1}.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_name)
        
        metadata = create_multi_document_pdf(pdf_path, doc_combo)
        dataset_metadata.append(metadata)
        
        print(f"Created {pdf_path} with {len(doc_combo)} documents")
    
    # Save overall dataset metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            'dataset_name': 'document_boundary_test_dataset',
            'created_date': random_date(),
            'pdf_count': count,
            'total_document_count': sum(len(meta['document_types']) for meta in dataset_metadata),
            'pdfs': dataset_metadata
        }, f, indent=2)
    
    print(f"Test dataset created with {count} PDFs in {output_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test dataset with known document boundaries")
    parser.add_argument("--output-dir", type=str, default="tests/fixtures/boundary_test_dataset",
                      help="Directory to save the test dataset")
    parser.add_argument("--count", type=int, default=5,
                      help="Number of test PDFs to generate")
    
    args = parser.parse_args()
    
    generate_test_dataset(args.output_dir, args.count)