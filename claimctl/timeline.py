"""Timeline extraction module for identifying and managing event chronology.

This module provides functionality to:
1. Extract timeline events from document chunks during ingestion
2. Generate comprehensive claim timelines from extracted events
3. Identify key documents and events based on importance and relevance
4. Track financial impacts of events and calculate running totals
5. Detect and highlight contradictions between events
"""

import re
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .config import get_config
from .database import (
    Document, 
    Page, 
    PageChunk, 
    TimelineEvent, 
    FinancialEvent,
    get_document_by_path,
    get_session, 
    get_timeline_events,
    save_timeline_event,
    save_financial_event,
    update_running_totals,
    get_financial_events,
    get_financial_summary,
    identify_contradictions,
    save_contradiction,
)
from .utils import console

# Define timeline event types
class EventType(str, Enum):
    """Timeline event types."""
    PROJECT_START = "project_start"
    PROJECT_COMPLETION = "project_completion"
    CHANGE_ORDER = "change_order"
    DELAY = "delay"
    PAYMENT = "payment"
    NOTICE = "notice"
    CLAIM = "claim"
    DISPUTE = "dispute"
    AGREEMENT = "agreement"
    CORRESPONDENCE = "correspondence"
    SUBMITTAL = "submittal"
    RFI = "request_for_information"
    MEETING = "meeting"
    OTHER = "other"

# Define financial impact types
class FinancialImpactType(str, Enum):
    """Financial impact types."""
    CHANGE_ORDER = "change_order"
    PAYMENT = "payment"
    CLAIM = "claim"
    DELAY_COST = "delay_cost"
    CREDIT = "credit"
    FEE = "fee"
    PENALTY = "penalty"
    OTHER = "other"

# Define contradiction types
class ContradictionType(str, Enum):
    """Types of contradictions between events."""
    DATE = "date"
    FINANCIAL = "financial"
    SCOPE = "scope"
    RESPONSIBILITY = "responsibility"
    OTHER = "other"


# Event extraction prompt
TIMELINE_EXTRACTION_PROMPT = """
You are an expert construction claim timeline analyzer. Analyze this document chunk and determine if it contains significant timeline events related to a construction claim. 

INSTRUCTIONS:
1. Read the document chunk carefully
2. Identify if the document contains any significant timeline events
3. Extract any relevant dates, events, and descriptions
4. Assign an event type and importance score to each event
5. Extract financial impact information if present
6. Identify potential contradictions with other events
7. Provide a confidence score for your extraction
8. If no significant timeline events are found, respond with "NO_EVENTS"

DOCUMENT CHUNK:
{text}

Document Type: {chunk_type}
Date: {doc_date}
Document ID: {doc_id}
Project: {project_name}
Page Number: {page_num}

If the chunk contains significant timeline events, format each event using this JSON structure:
```json
{{
  "event_date": "YYYY-MM-DD", 
  "event_type": "[project_start|project_completion|change_order|delay|payment|notice|claim|dispute|agreement|correspondence|submittal|request_for_information|meeting|other]",
  "description": "Brief description of the event",
  "importance_score": "<float between 0-1, where 1 is highest importance>",
  "confidence": "<float between 0-1 indicating confidence in this extraction>",
  "referenced_documents": "Any other documents referenced",
  "involved_parties": "Any parties mentioned as involved",
  "financial_impact": {{
    "amount": "<numeric amount, e.g. 25000>",
    "is_additive": "<boolean: true if this adds to project cost, false if it reduces cost>",
    "currency": "<currency code, default to USD>",
    "description": "<description of financial impact>",
    "impact_type": "[change_order|payment|claim|delay_cost|credit|fee|penalty|other]"
  }},
  "potential_contradiction": {{
    "description": "<description of potential contradiction with other documents>",
    "contradiction_type": "[date|financial|scope|responsibility|other]"
  }}
}}
```

Financial impact guidelines:
- Include financial impact information ONLY if explicitly mentioned in the document
- Extract the exact amount; do not make assumptions
- Set is_additive to true for costs that increase the project value, false for credits that reduce value
- Use the appropriate impact_type based on the context

Contradiction detection guidelines:
- Note potential contradictions only when document explicitly indicates conflict with other documents
- Focus on contradictions in dates, financial amounts, scope of work, or responsibility assignment
- Provide specific details about the nature of the contradiction

If multiple events are found, return an array of JSON objects.
If NO significant timeline events are found, respond with exactly: "NO_EVENTS"
"""


def extract_timeline_events(
    chunk: Dict[str, Any],
    matter_id: int,
    batch_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Extract timeline events from a document chunk.
    
    Args:
        chunk: Document chunk data
        matter_id: ID of the matter
        batch_mode: If True, reduces logging for batch processing
        
    Returns:
        List of extracted timeline events
    """
    config = get_config()
    
    # Skip extraction for document types that are unlikely to contain timeline events
    low_value_types = ["Photo", "Drawing", "Other"]
    if chunk.get("chunk_type") in low_value_types and batch_mode:
        return []
    
    try:
        # Get chunk information for the prompt
        chunk_text = chunk.get("text", "")
        chunk_type = chunk.get("chunk_type", "Unknown")
        doc_date = chunk.get("doc_date", "Unknown")
        doc_id = chunk.get("doc_id", "")
        project_name = chunk.get("project_name", "")
        page_num = chunk.get("page_num", "")
        
        # Prepare prompt
        prompt = TIMELINE_EXTRACTION_PROMPT.format(
            text=chunk_text[:3000],  # Limit text to 3000 chars to stay within token limits
            chunk_type=chunk_type,
            doc_date=doc_date,
            doc_id=doc_id,
            project_name=project_name,
            page_num=page_num,
        )
        
        # Query OpenAI
        client = OpenAI(api_key=config.openai.API_KEY)
        response = client.chat.completions.create(
            model=config.openai.MODEL,  # Use the configured model instead of hardcoded value
            messages=[
                {
                    "role": "system",
                    "content": "You are a construction claim timeline analyzer that extracts timeline events from documents.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "text"},
            temperature=0.2,  # Lower temperature for more factual responses
        )
        
        result = response.choices[0].message.content.strip()
        
        # Check if no events were found
        if result == "NO_EVENTS":
            if not batch_mode:
                console.log("[yellow]No timeline events found in this document chunk")
            return []
        
        # Parse JSON response
        import json
        import re
        
        # Extract JSON from the response (handling potential markdown code blocks)
        json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try finding any JSON array or object
            json_match = re.search(r'(\[.*\]|\{.*\})', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = result
        
        # Parse the JSON
        try:
            events_data = json.loads(json_str)
            # Handle both single event and array of events
            if isinstance(events_data, dict):
                events_data = [events_data]
        except json.JSONDecodeError as e:
            if not batch_mode:
                console.log(f"[bold red]Error parsing timeline events: {str(e)}")
                console.log(f"[dim yellow]Raw response: {result}")
            return []
        
        # Get document information for the event records
        document = None
        with get_session() as session:
            page = session.query(Page).filter(Page.id == chunk.get("page_id")).first()
            if page:
                document = session.query(Document).filter(Document.id == page.document_id).first()
        
        if not document:
            if not batch_mode:
                console.log("[bold red]Error: Could not find document for this chunk")
            return []
        
        # Process each event
        timeline_events = []
        for event_data in events_data:
            # Parse date
            event_date = None
            if event_data.get("event_date"):
                try:
                    event_date = datetime.strptime(event_data["event_date"], "%Y-%m-%d").date()
                except ValueError:
                    # If date parsing fails, use document date
                    if isinstance(chunk.get("doc_date"), date):
                        event_date = chunk["doc_date"]
                    elif isinstance(chunk.get("doc_date"), str):
                        try:
                            event_date = datetime.strptime(chunk["doc_date"], "%Y-%m-%d").date()
                        except ValueError:
                            event_date = None
            
            # Only use valid event types
            event_type = event_data.get("event_type", "other")
            if event_type not in [e.value for e in EventType]:
                event_type = "other"
            
            # Extract financial impact data if available
            financial_impact = None
            financial_impact_description = None
            financial_impact_type = None
            
            if event_data.get("financial_impact"):
                fin_data = event_data["financial_impact"]
                
                # Parse financial amount
                try:
                    financial_impact = float(fin_data.get("amount", 0))
                    
                    # Apply sign based on is_additive flag
                    if fin_data.get("is_additive") == False:
                        financial_impact = -financial_impact
                        
                    financial_impact_description = fin_data.get("description", "")
                    impact_type = fin_data.get("impact_type", "other")
                    
                    # Validate impact type
                    if impact_type not in [t.value for t in FinancialImpactType]:
                        impact_type = "other"
                        
                    financial_impact_type = impact_type
                except (ValueError, TypeError):
                    # If we can't parse the amount, skip financial impact
                    financial_impact = None
            
            # Extract contradiction data if available
            has_contradiction = False
            contradiction_details = None
            
            if event_data.get("potential_contradiction"):
                contra_data = event_data["potential_contradiction"]
                has_contradiction = True
                contradiction_details = contra_data.get("description", "")
            
            # Create timeline event record
            timeline_event = {
                "matter_id": matter_id,
                "chunk_id": chunk.get("id"),
                "document_id": document.id,
                "event_date": event_date,
                "event_type": event_type,
                "description": event_data.get("description", ""),
                "importance_score": float(event_data.get("importance_score", 0.5)),
                "confidence": float(event_data.get("confidence", 0.5)),
                "referenced_documents": event_data.get("referenced_documents", ""),
                "involved_parties": event_data.get("involved_parties", ""),
                "has_contradiction": has_contradiction,
                "contradiction_details": contradiction_details,
                "financial_impact": financial_impact,
                "financial_impact_description": financial_impact_description,
                "financial_impact_type": financial_impact_type,
            }
            
            # Save to database
            event_id = save_timeline_event(timeline_event)
            if event_id:
                timeline_event["id"] = event_id
                timeline_events.append(timeline_event)
                
                # If there's financial impact data, create a financial event
                if financial_impact is not None and event_data.get("financial_impact"):
                    fin_data = event_data["financial_impact"]
                    
                    # Create financial event
                    financial_event = {
                        "matter_id": matter_id,
                        "timeline_event_id": event_id,
                        "document_id": document.id,
                        "amount": abs(financial_impact),  # Store absolute amount with sign in is_additive
                        "amount_description": financial_impact_description,
                        "currency": fin_data.get("currency", "USD"),
                        "event_type": financial_impact_type,
                        "category": event_type,  # Use timeline event type as category
                        "event_date": event_date,
                        "is_additive": financial_impact >= 0,  # Determine from sign
                        "confidence": float(event_data.get("confidence", 0.5)),
                    }
                    
                    # Save financial event
                    fin_event_id = save_financial_event(financial_event)
                    if not batch_mode and fin_event_id:
                        console.log(f"[green]Financial event saved (ID: {fin_event_id}, Amount: ${abs(financial_impact):.2f})")
                
                if not batch_mode:
                    console.log(f"[green]Timeline event extracted and saved (ID: {event_id})")
            else:
                if not batch_mode:
                    console.log("[bold red]Error saving timeline event")
        
        # If financial events were created, update running totals
        if any(e.get("financial_impact") is not None for e in timeline_events):
            update_running_totals(matter_id)
            
        return timeline_events
        
    except Exception as e:
        if not batch_mode:
            console.log(f"[bold red]Error extracting timeline events: {str(e)}")
        return []


def batch_extract_timeline_events(
    chunks: List[Dict[str, Any]],
    matter_id: int,
    progress: Optional[Progress] = None,
) -> List[Dict[str, Any]]:
    """Extract timeline events from multiple document chunks.
    
    Args:
        chunks: List of document chunks
        matter_id: ID of the matter
        progress: Optional progress bar
        
    Returns:
        List of extracted timeline events
    """
    all_events = []
    
    # Create a task in the progress bar if provided
    task_id = None
    if progress is not None:
        task_id = progress.add_task("Extracting timeline events", total=len(chunks))
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Update progress
        if progress is not None:
            progress.update(task_id, advance=0, description=f"Processing chunk {i+1}/{len(chunks)}")
        
        # Extract timeline events
        events = extract_timeline_events(chunk, matter_id, batch_mode=True)
        all_events.extend(events)
        
        # Update progress
        if progress is not None:
            progress.update(task_id, advance=1)
    
    return all_events


def generate_claim_timeline(
    matter_id: int,
    event_types: Optional[List[str]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    importance_threshold: float = 0.3,
    confidence_threshold: float = 0.5,
    include_financial_impacts: bool = True,
    include_contradictions: bool = True,
    detect_new_contradictions: bool = True,
    max_events: int = 100,
) -> Dict[str, Any]:
    """Generate a comprehensive timeline for a claim.
    
    Args:
        matter_id: ID of the matter
        event_types: Optional list of event types to include
        date_from: Optional start date
        date_to: Optional end date
        importance_threshold: Minimum importance score for events (0-1)
        confidence_threshold: Minimum confidence score for events (0-1)
        include_financial_impacts: Whether to include financial impact details
        include_contradictions: Whether to include contradiction details
        detect_new_contradictions: Whether to run contradiction detection
        max_events: Maximum number of events to include
        
    Returns:
        Dictionary containing timeline data
    """
    # Get timeline events from database
    events = get_timeline_events(
        matter_id=matter_id,
        event_types=event_types,
        date_from=date_from,
        date_to=date_to,
        min_confidence=confidence_threshold,
        min_importance=importance_threshold,
        include_contradictions=include_contradictions,
        include_financial_impacts=include_financial_impacts,
        limit=max_events,
        sort_by="date",
    )
    
    if not events:
        console.log("[yellow]No timeline events found for this matter")
        return {"events": [], "summary": "No timeline events found."}
    
    # Group events by month
    events_by_month = {}
    for event in events:
        if event.get("event_date"):
            date_obj = datetime.fromisoformat(event["event_date"]).date()
            month_key = f"{date_obj.year}-{date_obj.month:02d}"
            
            if month_key not in events_by_month:
                events_by_month[month_key] = []
            
            events_by_month[month_key].append(event)
    
    # Generate timeline summary using LLM
    summary = generate_timeline_summary(events)
    
    # Add financial summary if requested
    financial_summary = None
    if include_financial_impacts:
        financial_summary = get_financial_summary(
            matter_id=matter_id,
            event_types=event_types,
            date_from=date_from,
            date_to=date_to,
        )
    
    # Identify contradictions if requested
    contradictions = []
    if include_contradictions and detect_new_contradictions:
        contradictions = identify_contradictions(
            matter_id=matter_id,
            min_confidence=confidence_threshold,
            max_date_diff_days=30,
        )
        
        # Save any newly identified contradictions to the database
        for contradiction in contradictions:
            if "event1_id" in contradiction and "event2_id" in contradiction:
                save_contradiction(
                    event1_id=contradiction["event1_id"],
                    event2_id=contradiction["event2_id"],
                    contradiction_details=contradiction["details"],
                )
    
    # Organize results
    timeline_data = {
        "events": events,
        "events_by_month": events_by_month,
        "summary": summary,
        "total_events": len(events),
        "financial_summary": financial_summary,
        "contradictions": contradictions,
    }
    
    return timeline_data


def generate_timeline_summary(events: List[Dict[str, Any]]) -> str:
    """Generate a summary of the timeline using the LLM.
    
    Args:
        events: List of timeline events
        
    Returns:
        Summary text
    """
    config = get_config()
    
    if not events:
        return "No timeline events found."
    
    # Format events for the prompt
    events_text = ""
    for i, event in enumerate(events[:20]):  # Limit to 20 events for the prompt
        events_text += f"Event {i+1}: {event.get('event_date', 'Unknown date')} - {event.get('event_type')}\n"
        events_text += f"Description: {event.get('description')}\n"
        
        # Add financial impact information if available
        if "financial_impact" in event and event["financial_impact"]:
            amount = event["financial_impact"]
            impact_type = event.get("financial_impact_type", "unknown")
            impact_desc = event.get("financial_impact_description", "")
            events_text += f"Financial Impact: ${abs(amount):,.2f} ({impact_type}) - {impact_desc}\n"
        
        # Add contradiction information if available
        if event.get("has_contradiction", False):
            events_text += f"Contradiction: {event.get('contradiction_details', 'Potential contradiction detected')}\n"
        
        events_text += f"Importance: {event.get('importance_score')}\n"
        events_text += f"Document: {event.get('document', {}).get('file_name', 'Unknown')}\n\n"
    
    # Calculate financial totals for the prompt
    total_financial_impact = 0
    financial_events_count = 0
    for event in events:
        if "financial_impact" in event and event["financial_impact"] is not None:
            total_financial_impact += event["financial_impact"]
            financial_events_count += 1
    
    # Count contradictions
    contradiction_count = sum(1 for event in events if event.get("has_contradiction", False))
    
    # Add financial summary and contradiction info to the prompt
    financial_summary_text = ""
    if financial_events_count > 0:
        financial_summary_text = f"""
        Financial Summary:
        - Total financial impact: ${total_financial_impact:,.2f}
        - Number of financial events: {financial_events_count}
        - Average financial impact per event: ${total_financial_impact / financial_events_count:,.2f}
        """
    
    contradiction_text = ""
    if contradiction_count > 0:
        contradiction_text = f"""
        Contradictions:
        - Number of events with contradictions: {contradiction_count}
        - Pay special attention to contradictory information in the timeline.
        """
    
    # Create summary prompt
    prompt = f"""
    You are a construction claim analyst summarizing a timeline of events. Below are the key events from a construction claim timeline.
    Please provide a concise executive summary (3-5 paragraphs) highlighting the most significant events, patterns, and their potential significance to the claim.
    
    Focus on:
    1. Project timeline overview (start, completion, key milestones)
    2. Critical delay events and their potential impacts
    3. Formal notices or claims and their timing
    4. Payment issues or disputes
    5. Financial impacts and their cumulative effect
    6. Any contradictions or inconsistencies in the timeline
    7. Any pattern of events that might indicate liability or entitlement
    
    Timeline Events:
    {events_text}
    
    {financial_summary_text}
    
    {contradiction_text}
    
    Generate a concise executive summary of this timeline, highlighting key events and patterns relevant to a construction claim analysis.
    Your summary should be factual and neutral, avoiding speculation beyond what is supported by the timeline.
    Include a brief analysis of the financial impacts and any notable contradictions.
    """
    
    try:
        # Query OpenAI
        client = OpenAI(api_key=config.openai.API_KEY)
        response = client.chat.completions.create(
            model=config.openai.MODEL,  # Use the configured model instead of hardcoded value
            messages=[
                {
                    "role": "system",
                    "content": "You are a construction claim timeline analyst.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,  # Lower temperature for more factual responses
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        console.log(f"[bold red]Error generating timeline summary: {str(e)}")
        return "Error generating timeline summary."


def display_timeline(timeline_data: Dict[str, Any], format: str = "table") -> None:
    """Display a timeline in the specified format.
    
    Args:
        timeline_data: Timeline data from generate_claim_timeline
        format: Output format ('table' or 'text')
    """
    events = timeline_data.get("events", [])
    summary = timeline_data.get("summary", "")
    financial_summary = timeline_data.get("financial_summary")
    contradictions = timeline_data.get("contradictions", [])
    
    if not events:
        console.print("[yellow]No timeline events to display")
        return
    
    # Display summary
    console.rule("[bold blue]Timeline Summary")
    console.print(summary)
    console.print()
    
    # Display financial summary if available
    if financial_summary:
        console.rule("[bold green]Financial Impact Summary")
        console.print(f"Total Financial Impact: [bold]${financial_summary.get('total_amount', 0):,.2f}[/bold]")
        console.print(f"Positive Impacts: [green]${financial_summary.get('total_positive', 0):,.2f}[/green]")
        console.print(f"Negative Impacts: [red]${financial_summary.get('total_negative', 0):,.2f}[/red]")
        console.print(f"Financial Events: [blue]{financial_summary.get('event_count', 0)}[/blue]")
        
        # Display category breakdown
        if financial_summary.get('event_type_totals'):
            console.print("\nImpact by Event Type:")
            for event_type, amount in sorted(financial_summary['event_type_totals'].items(), key=lambda x: abs(x[1]), reverse=True):
                color = "green" if amount >= 0 else "red"
                console.print(f"  {event_type}: [{color}]${amount:,.2f}[/{color}]")
                
        # Display monthly totals
        if financial_summary.get('monthly_totals'):
            console.print("\nMonthly Financial Impact:")
            for month, amount in sorted(financial_summary['monthly_totals'].items()):
                year, month_num = month.split('-')
                month_name = datetime(int(year), int(month_num), 1).strftime("%B %Y")
                color = "green" if amount >= 0 else "red"
                console.print(f"  {month_name}: [{color}]${amount:,.2f}[/{color}]")
        
        console.print()
        
    # Display contradictions if any
    if contradictions:
        console.rule("[bold red]Contradictions Detected")
        console.print(f"Detected {len(contradictions)} contradictions in timeline events.")
        
        for i, contradiction in enumerate(contradictions[:5], 1):  # Limit to top 5
            event1_date = contradiction.get("event1_date", "Unknown")
            event2_date = contradiction.get("event2_date", "Unknown")
            event_type = contradiction.get("event_type", "Unknown")
            
            panel = Panel(
                Text(contradiction.get("details", ""), style="yellow"),
                title=f"Contradiction {i}: {event_type}",
                subtitle=f"Events from {event1_date} and {event2_date}",
                border_style="red"
            )
            console.print(panel)
        
        if len(contradictions) > 5:
            console.print(f"[dim]... and {len(contradictions) - 5} more contradictions[/dim]")
            
        console.print()
    
    # Display events
    if format == "table":
        table = Table(title="Claim Timeline")
        table.add_column("Date", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Financial", style="yellow", justify="right")
        table.add_column("Document", style="blue")
        table.add_column("Flags", style="red")
        
        for event in events:
            event_date = event.get("event_date", "Unknown")
            event_type = event.get("event_type", "other")
            description = event.get("description", "")
            
            # Truncate description if too long
            if len(description) > 50:
                description = description[:47] + "..."
            
            document_name = event.get("document", {}).get("file_name", "Unknown")
            
            # Format financial impact
            financial_impact = ""
            if "financial_impact" in event and event["financial_impact"] is not None:
                amount = event["financial_impact"]
                if amount >= 0:
                    financial_impact = f"[green]${amount:,.2f}[/green]"
                else:
                    financial_impact = f"[red]${amount:,.2f}[/red]"
            
            # Format flags (contradictions)
            flags = ""
            if event.get("has_contradiction", False):
                flags = "[bold red]⚠ Contradiction[/bold red]"
            
            table.add_row(
                event_date,
                event_type,
                description,
                financial_impact,
                document_name,
                flags,
            )
        
        console.print(table)
    else:
        # Text format
        console.rule("[bold blue]Timeline Events")
        
        current_year = None
        current_month = None
        
        for event in events:
            if event.get("event_date"):
                date_obj = datetime.fromisoformat(event["event_date"]).date()
                year = date_obj.year
                month = date_obj.month
                
                # Print year/month headers
                if year != current_year:
                    console.print(f"\n[bold cyan]{year}")
                    current_year = year
                    current_month = None
                
                if month != current_month:
                    month_name = date_obj.strftime("%B")
                    console.print(f"[bold blue]  {month_name}")
                    current_month = month
                
                # Format financial impact string
                financial_str = ""
                if "financial_impact" in event and event["financial_impact"] is not None:
                    amount = event["financial_impact"]
                    if amount >= 0:
                        financial_str = f" [green](+${amount:,.2f})[/green]"
                    else:
                        financial_str = f" [red](${amount:,.2f})[/red]"
                
                # Format contradiction flag
                contradiction_str = ""
                if event.get("has_contradiction", False):
                    contradiction_str = " [bold red][⚠ Contradiction][/bold red]"
                
                # Print event
                event_date = date_obj.strftime("%d %b")
                console.print(f"[cyan]    {event_date}[/cyan] [magenta]{event.get('event_type')}[/magenta]{financial_str}{contradiction_str}")
                console.print(f"      [green]{event.get('description')}[/green]")
                
                # Print contradiction details if present
                if event.get("has_contradiction", False) and event.get("contradiction_details"):
                    console.print(f"      [red]Contradiction: {event.get('contradiction_details')}[/red]")
                
                console.print(f"      [dim]Document: {event.get('document', {}).get('file_name', 'Unknown')}[/dim]")
                console.print("")
            else:
                # Handle events without dates
                console.print(f"[magenta]{event.get('event_type')}[/magenta]")
                console.print(f"  [green]{event.get('description')}[/green]")
                console.print(f"  [dim]Document: {event.get('document', {}).get('file_name', 'Unknown')}[/dim]")
                console.print("")
    
    # Print statistics
    console.print(f"[dim]Total events: {len(events)}[/dim]")
    
    # Show financial events statistics if available
    if financial_summary:
        console.print(f"[dim]Financial events: {financial_summary.get('event_count', 0)}[/dim]")
    
    # Show contradiction statistics
    if timeline_data.get("contradictions"):
        console.print(f"[dim]Contradictions: {len(timeline_data['contradictions'])}[/dim]")


def extract_events_from_all_documents(
    matter_id: int,
    progress: Optional[Progress] = None,
    resume: bool = True,
    force: bool = False,
) -> int:
    """Extract timeline events from all documents in a matter.
    
    Args:
        matter_id: ID of the matter
        progress: Optional progress bar
        resume: Whether to resume from where the previous extraction left off
        force: Whether to force re-extraction of all events (ignores resume)
        
    Returns:
        Number of events extracted
    """
    # Get the matter info for the resume log
    with get_session() as session:
        matter = session.query(Matter).filter(Matter.id == matter_id).first()
        if not matter:
            console.log("[bold red]Matter not found")
            return 0
            
        matter_name = matter.name
    
    # Create resume log path
    config = get_config()
    matter_dir = Path(matter.data_directory).parent
    logs_dir = matter_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    resume_log_path = logs_dir / "timeline_extract_resume.log"
    
    # Get completed chunk IDs from resume log
    completed_chunk_ids = set()
    if resume and resume_log_path.exists() and not force:
        try:
            with open(resume_log_path, "r") as f:
                completed_chunk_ids = set(int(line.strip()) for line in f.readlines() if line.strip().isdigit())
            console.log(f"[bold yellow]Found {len(completed_chunk_ids)} previously processed chunks")
        except Exception as e:
            console.log(f"[bold red]Error reading resume log: {str(e)}")
    
    # If forcing re-extraction, clear existing events first
    if force:
        console.log("[bold yellow]Force mode enabled, deleting existing timeline events")
        with get_session() as session:
            # Get all timeline event IDs for this matter
            event_ids = [e.id for e in session.query(TimelineEvent.id).filter(TimelineEvent.matter_id == matter_id).all()]
            
            # Delete related financial events first (avoid foreign key constraint errors)
            if event_ids:
                session.query(FinancialEvent).filter(FinancialEvent.timeline_event_id.in_(event_ids)).delete(synchronize_session=False)
                session.query(TimelineEvent).filter(TimelineEvent.matter_id == matter_id).delete(synchronize_session=False)
                session.commit()
                
                # Clear resume log if it exists
                if resume_log_path.exists():
                    resume_log_path.unlink()
                
                console.log(f"[bold green]Deleted existing timeline events for matter '{matter_name}'")
    
    # Get all chunks for the matter
    with get_session() as session:
        # Get matter documents
        documents = session.query(Document).filter(Document.matter_id == matter_id).all()
        
        if not documents:
            console.log("[yellow]No documents found for this matter")
            return 0
        
        chunks = []
        for document in documents:
            # Get pages for document
            pages = session.query(Page).filter(Page.document_id == document.id).all()
            
            for page in pages:
                # Get chunks for page
                page_chunks = session.query(PageChunk).filter(PageChunk.page_id == page.id).all()
                
                for chunk in page_chunks:
                    # Skip already processed chunks if resuming and not forcing
                    if resume and not force and chunk.id in completed_chunk_ids:
                        continue
                        
                    chunks.append({
                        "id": chunk.id,
                        "text": chunk.text,
                        "page_id": chunk.page_id,
                        "chunk_type": chunk.chunk_type,
                        "doc_date": chunk.doc_date,
                        "doc_id": chunk.doc_id,
                        "project_name": chunk.project_name,
                        "page_num": page.page_num,
                    })
    
    # If all chunks have been processed, we're done
    if not chunks:
        if completed_chunk_ids:
            console.log("[bold green]All chunks have already been processed. Use --force to re-extract all events.")
        else:
            console.log("[yellow]No chunks to process")
        return 0
    
    # Process chunks in batches
    batch_size = 50
    total_events = 0
    
    # Use existing progress bar or create new one
    progress_owner = False
    if progress is None:
        from rich.progress import Progress as RichProgress
        progress = RichProgress()
        progress.start()
        progress_owner = True
    
    try:
        # Create overall task
        overall_task = progress.add_task(f"Processing {len(chunks)} chunks", total=len(chunks))
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Update progress description
            progress.update(
                overall_task,
                advance=0,
                description=f"Batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}"
            )
            
            # Process batch
            events = batch_extract_timeline_events(batch, matter_id, progress)
            total_events += len(events)
            
            # Update resume log with completed chunk IDs
            with open(resume_log_path, "a") as f:
                for chunk in batch:
                    f.write(f"{chunk['id']}\n")
            
            # Update overall progress
            progress.update(overall_task, advance=len(batch))
    
    finally:
        # Stop progress bar if we created it
        if progress_owner:
            progress.stop()
    
    return total_events