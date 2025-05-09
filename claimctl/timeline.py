"""Timeline extraction module for identifying and managing event chronology.

This module provides functionality to:
1. Extract timeline events from document chunks during ingestion
2. Generate comprehensive claim timelines from extracted events
3. Identify key documents and events based on importance and relevance
4. Track financial impacts of events and calculate running totals
5. Detect and highlight contradictions between events
"""

import re
import time
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from openai import OpenAI, OpenAIError, APITimeoutError, RateLimitError
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
    Matter,
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
You are an expert construction claim timeline analyzer. Analyze this document chunk efficiently and determine if it contains significant timeline events related to a construction claim. 

INSTRUCTIONS (PRIORITIZED):
1. FIRST, quickly scan for dates, monetary amounts, and key terms like "delay," "change order," or "claim"
2. If none found, immediately respond with "NO_EVENTS" without further analysis
3. If potential events found, extract relevant dates, descriptions, and event types
4. Assign event type and importance score only for significant events
5. Extract financial impact ONLY when explicitly mentioned with amounts
6. Only look for contradictions when document explicitly references conflicts
7. Work efficiently - focus on key facts rather than exhaustive analysis

DOCUMENT CHUNK:
{text}

Document Type: {chunk_type}
Date: {doc_date}
Document ID: {doc_id}
Project: {project_name}
Page Number: {page_num}

For significant timeline events ONLY, use this JSON structure:
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
- Include financial impact information ONLY if explicitly mentioned with a clear amount
- Set is_additive to true for costs that increase the project value, false for credits

Contradiction detection guidelines:
- Only include contradictions when document explicitly mentions conflicts
- Be very selective about marking contradictions

If multiple events are found, return an array of JSON objects.
If NO significant timeline events are found, respond with exactly: "NO_EVENTS"
"""


def extract_timeline_events(
    chunk: Dict[str, Any],
    matter_id: int,
    batch_mode: bool = False,
    document_context: bool = False,
) -> List[Dict[str, Any]]:
    """Extract timeline events from a document chunk.
    
    Args:
        chunk: Document chunk data
        matter_id: ID of the matter
        batch_mode: If True, reduces logging for batch processing
        document_context: If True, the chunk contains additional document context
        
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
        
        # Prepare prompt with additional context if available
        if document_context and "enhanced_context" in chunk:
            # Include enhanced document context in the prompt
            enhanced_text = f"{chunk_text}\n\n{chunk['enhanced_context']}"
            
            # Log the use of enhanced context
            if not batch_mode:
                console.log("[green]Using enhanced document context for extraction")
                
            prompt = TIMELINE_EXTRACTION_PROMPT.format(
                text=enhanced_text[:3800],  # Increased limit to include context, still within token limits
                chunk_type=chunk_type,
                doc_date=doc_date,
                doc_id=doc_id,
                project_name=project_name,
                page_num=page_num,
            )
        else:
            # Standard prompt without enhanced context
            prompt = TIMELINE_EXTRACTION_PROMPT.format(
                text=chunk_text[:3000],  # Limit text to 3000 chars to stay within token limits
                chunk_type=chunk_type,
                doc_date=doc_date,
                doc_id=doc_id,
                project_name=project_name,
                page_num=page_num,
            )
        
        # Log model being used for debugging
        if not batch_mode:
            console.log(f"[dim]Using model: {config.openai.MODEL}")
        
        # Query OpenAI with retry logic
        client = OpenAI(api_key=config.openai.API_KEY)
        
        # Create log dir for API responses if needed
        debug_log_dir = Path("./logs/api_debug")
        debug_log_dir.mkdir(exist_ok=True, parents=True)
        
        start_time = datetime.now()
        api_log_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{chunk.get('id')}"
        
        # Save request for debugging
        with open(debug_log_dir / f"request_{api_log_id}.txt", "w") as f:
            f.write(f"Model: {config.openai.MODEL}\n")
            f.write(f"Time: {start_time}\n")
            f.write(f"Chunk ID: {chunk.get('id')}\n")
            f.write(f"Document Type: {chunk_type}\n")
            f.write("System message:\n")
            f.write("You are a construction claim timeline analyzer that extracts timeline events from documents.\n\n")
            f.write("User message:\n")
            f.write(prompt)
        
        # Retry parameters
        max_retries = 3
        retry_delay = 2  # seconds
        attempts = 0
        response = None
        
        try:
            # Retry loop for API calls
            while attempts < max_retries:
                attempts += 1
                try:
                    # Log request for debugging
                    if not batch_mode:
                        if attempts > 1:
                            console.log(f"[dim]Retry {attempts}/{max_retries} for chunk {chunk.get('id')}")
                        else:
                            console.log(f"[dim]Making API request for chunk {chunk.get('id')}")
                    
                    # Make API call
                    response = client.chat.completions.create(
                        model=config.openai.MODEL,  # Use the configured model instead of hardcoded value
                        reasoning_effort="low",  # Using low effort for timeline extraction to improve speed
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
                        response_format={"type": "text"}
                        # Removed temperature parameter as it's causing API errors with o4-mini
                    )
                    
                    # If we get here, the call succeeded
                    break
                    
                except (RateLimitError, APITimeoutError) as e:
                    # These errors are worth retrying with backoff
                    if attempts < max_retries:
                        wait_time = retry_delay * (2 ** (attempts - 1))  # Exponential backoff
                        console.log(f"[yellow]API rate limit or timeout error, retrying in {wait_time}s ({attempts}/{max_retries}): {str(e)}")
                        time.sleep(wait_time)
                    else:
                        # We've used all our retries, re-raise
                        raise
                        
                except Exception as e:
                    # Don't retry other types of errors
                    raise
            
            # Make sure we got a response
            if response is None:
                raise Exception("Failed to get response after retries")
                
            # Log API response time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Save successful response for debugging
            result = response.choices[0].message.content.strip()
            with open(debug_log_dir / f"response_{api_log_id}.txt", "w") as f:
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Attempts: {attempts}/{max_retries}\n")
                f.write("Response:\n")
                f.write(result)
                
            if not batch_mode:
                console.log(f"[dim]API response received in {duration:.2f}s after {attempts} attempt(s)")
                
        except Exception as api_error:
            # Log API error for debugging
            error_msg = f"API Error: {str(api_error)}"
            
            # Save error response
            with open(debug_log_dir / f"error_{api_log_id}.txt", "w") as f:
                f.write(f"Duration: {(datetime.now() - start_time).total_seconds():.2f}s\n")
                f.write(f"Attempts: {attempts}/{max_retries}\n")
                f.write("Error:\n")
                f.write(error_msg)
                f.write("\n\nChunk info:\n")
                f.write(f"Document: {chunk.get('file_name', 'Unknown')}\n")
                f.write(f"Page: {chunk.get('page_num', 'Unknown')}\n")
                f.write(f"Type: {chunk_type}\n")
            
            # Always print API errors to console for visibility regardless of batch mode
            console.log(f"[bold red]{error_msg}")
            console.log(f"[red]Document: {chunk.get('file_name', 'Unknown')}, Page: {chunk.get('page_num', 'Unknown')}")
            
            raise Exception(error_msg)
        
        # Process the content from the response
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
    error_count = 0
    
    # Create a task in the progress bar if provided
    task_id = None
    if progress is not None:
        task_id = progress.add_task("Extracting timeline events", total=len(chunks))
    
    # Get matter log directory for additional logging
    with get_session() as session:
        matter = session.query(Matter).filter(Matter.id == matter_id).first()
        if matter:
            matter_dir = Path(matter.data_directory).parent
            logs_dir = matter_dir / "logs"
            logs_dir.mkdir(exist_ok=True, parents=True)
            batch_log_path = logs_dir / "batch_timeline_extract.log"
            
            # Initialize batch log
            with open(batch_log_path, "a") as log:
                log.write(f"\n--- Batch extraction starting at {datetime.now()} ---\n")
                log.write(f"Processing {len(chunks)} chunks with model: {get_config().openai.MODEL}\n")
    
    # Process each chunk
    config = get_config()
    console.log(f"[bold blue]Timeline extraction using model: {config.openai.MODEL}")
    
    for i, chunk in enumerate(chunks):
        # Update progress
        if progress is not None:
            progress.update(task_id, advance=0, description=f"Processing chunk {i+1}/{len(chunks)}")
        
        # Extract detailed chunk info for logging
        chunk_info = f"Chunk {i+1}/{len(chunks)} - ID: {chunk.get('id')}, Type: {chunk.get('chunk_type')}"
        if matter:
            with open(batch_log_path, "a") as log:
                log.write(f"\nProcessing {chunk_info} - {datetime.now()}\n")
        
        try:
            # Extract timeline events
            start_time = datetime.now()
            events = extract_timeline_events(chunk, matter_id, batch_mode=True)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            all_events.extend(events)
            
            # Log results
            if matter:
                with open(batch_log_path, "a") as log:
                    log.write(f"  Completed in {duration:.2f}s - Found {len(events)} events\n")
            
            # Print periodic status updates
            if (i+1) % 10 == 0 or i == 0 or i == len(chunks)-1:
                console.log(f"[green]Processed {i+1}/{len(chunks)} chunks - Found {len(all_events)} events so far")
                
        except Exception as e:
            error_count += 1
            error_message = f"Error processing {chunk_info}: {str(e)}"
            
            # Always log errors to console regardless of batch mode
            console.log(f"[bold red]{error_message}")
            
            # Print more details to make debugging easier
            console.log(f"[red]Document: {chunk.get('file_name', 'Unknown')}, Page: {chunk.get('page_num', 'Unknown')}")
            
            if matter:
                with open(batch_log_path, "a") as log:
                    log.write(f"  ERROR: {str(e)}\n")
                    log.write(f"  Document: {chunk.get('file_name', 'Unknown')}, Page: {chunk.get('page_num', 'Unknown')}\n")
        
        # Update progress
        if progress is not None:
            progress.update(task_id, advance=1)
    
    # Log completion
    if matter:
        with open(batch_log_path, "a") as log:
            log.write(f"\n--- Batch extraction completed at {datetime.now()} ---\n")
            log.write(f"Total events extracted: {len(all_events)}\n")
            log.write(f"Total errors: {error_count}\n")
    
    console.log(f"[bold green]Batch extraction complete - Found {len(all_events)} events, encountered {error_count} errors")
    
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
            reasoning_effort="low",  # Using low effort for timeline generation to improve speed
            messages=[
                {
                    "role": "system",
                    "content": "You are a construction claim timeline analyst.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            # Removed temperature parameter as it's causing API errors with o4-mini
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
    document_aware: bool = True,
    parallel: bool = True,
    max_workers: int = 4,
) -> int:
    """Extract timeline events from all documents in a matter.
    
    Args:
        matter_id: ID of the matter
        progress: Optional progress bar
        resume: Whether to resume from where the previous extraction left off
        force: Whether to force re-extraction of all events (ignores resume)
        document_aware: Whether to process chunks with document context (groups chunks by document)
        parallel: Whether to process documents in parallel
        max_workers: Maximum number of parallel workers for document processing
        
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
    total_processed = 0
    total_api_calls = 0
    total_api_errors = 0
    
    # Create log file for detailed tracking
    log_file_path = logs_dir / "timeline_extract_detail.log"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"===============================================\n")
        log_file.write(f"Timeline extraction started at {datetime.now()}\n")
        log_file.write(f"Matter: {matter_name} (ID: {matter_id})\n")
        log_file.write(f"Total chunks to process: {len(chunks)}\n")
        log_file.write(f"Model: {get_config().openai.MODEL}\n")
        log_file.write(f"Document-aware mode: {document_aware}\n")
        log_file.write(f"===============================================\n\n")
    
    # Use existing progress bar or create new one
    progress_owner = False
    if progress is None:
        from rich.progress import Progress as RichProgress
        progress = RichProgress()
        progress.start()
        progress_owner = True
    
    try:
        # Create overall task
        if document_aware:
            # Group chunks by document for document-aware processing
            console.log("[bold green]Using document-aware processing to maintain context")
            
            # Create a dictionary to group chunks by document
            docs_to_chunks = {}
            for chunk in chunks:
                doc_id = chunk.get("doc_id", "unknown")
                # Use both doc_id and filename to ensure unique grouping
                doc_key = f"{doc_id}_{chunk.get('file_name', '')}"
                if doc_key not in docs_to_chunks:
                    docs_to_chunks[doc_key] = []
                docs_to_chunks[doc_key].append(chunk)
            
            # Sort chunks within each document by page_num and chunk_index
            for doc_key in docs_to_chunks:
                docs_to_chunks[doc_key].sort(
                    key=lambda x: (x.get("page_num", 0), x.get("chunk_index", 0))
                )
            
            console.log(f"[green]Grouped {len(chunks)} chunks into {len(docs_to_chunks)} documents")
            
            # Create a task for document processing
            overall_task = progress.add_task(f"Processing {len(docs_to_chunks)} documents", total=len(docs_to_chunks))
            
            # Process documents in parallel or sequentially
            if parallel and len(docs_to_chunks) > 1:
                import concurrent.futures
                from threading import Lock
                
                # Use a lock for thread-safe operations
                resume_log_lock = Lock()
                log_file_lock = Lock()
                stats_lock = Lock()
                
                # Initialize shared counters
                shared_stats = {
                    "total_events": 0,
                    "total_processed": 0,
                    "total_api_calls": 0,
                    "total_api_errors": 0
                }
                
                # Function to process a single document with all its chunks
                def process_document(doc_num, doc_info):
                    doc_key, doc_chunks = doc_info
                    
                    if not doc_chunks:
                        return 0, 0, 0, 0  # No events, processed, API calls, errors
                    
                    # Extract document info
                    doc_name = doc_chunks[0].get("file_name", "Unknown")
                    doc_pages = max([c.get("page_num", 0) for c in doc_chunks])
                    
                    # Update console (thread-safe)
                    console.log(f"[bold blue]Processing document {doc_num}/{len(docs_to_chunks)}: {doc_name} ({doc_pages} pages)")
                    
                    # Log processing start (thread-safe)
                    with log_file_lock:
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"Document {doc_num}/{len(docs_to_chunks)} - {doc_name} started at {datetime.now()}\n")
                            log_file.write(f"  Contains {len(doc_chunks)} chunks across {doc_pages} pages\n")
                    
                    # Process document chunks
                    doc_batch_size = min(5, len(doc_chunks))  # Use smaller batches for document processing
                    doc_events = 0
                    doc_errors = 0
                    doc_api_calls = 0
                    doc_chunks_processed = 0
                    doc_start_time = datetime.now()
                    
                    # Process groups of chunks with document context
                    for i in range(0, len(doc_chunks), doc_batch_size):
                        context_batch = doc_chunks[i:i+doc_batch_size]
                        
                        for j, chunk in enumerate(context_batch):
                            chunk_start_time = datetime.now()
                            try:
                                # Log extract attempt (thread-safe)
                                with log_file_lock:
                                    with open(log_file_path, "a") as log_file:
                                        log_file.write(f"  Chunk {j+1}/{len(context_batch)} (ID: {chunk['id']}) - Processing at {chunk_start_time}\n")
                                
                                # Add document context to the chunk for enhanced extraction
                                if len(context_batch) > 1:
                                    # Add surrounding context to the chunk
                                    chunk_with_context = chunk.copy()
                                    
                                    # Include information about surrounding chunks
                                    context_info = f"\nDocument context: This is chunk {j+1} of {len(context_batch)} from page {chunk.get('page_num', 'unknown')}.\n"
                                    
                                    # Add previous chunk summary if available
                                    if j > 0:
                                        prev_chunk = context_batch[j-1]
                                        context_info += f"\nPrevious chunk from page {prev_chunk.get('page_num', 'unknown')} begins with: {prev_chunk.get('text', '')[:150]}...\n"
                                    
                                    # Add next chunk summary if available
                                    if j < len(context_batch) - 1:
                                        next_chunk = context_batch[j+1]
                                        context_info += f"\nNext chunk from page {next_chunk.get('page_num', 'unknown')} begins with: {next_chunk.get('text', '')[:150]}...\n"
                                    
                                    # Append context to chunk text
                                    chunk_with_context["enhanced_context"] = context_info
                                else:
                                    chunk_with_context = chunk
                                
                                # Process chunk with context
                                doc_api_calls += 1
                                chunk_events = extract_timeline_events(chunk_with_context, matter_id, batch_mode=True, document_context=True)
                                doc_events += len(chunk_events)
                                
                                # Log result (thread-safe)
                                chunk_end_time = datetime.now()
                                duration = (chunk_end_time - chunk_start_time).total_seconds()
                                with log_file_lock:
                                    with open(log_file_path, "a") as log_file:
                                        log_file.write(f"    Completed in {duration:.2f}s - Found {len(chunk_events)} events\n")
                                
                            except Exception as e:
                                # Log error (thread-safe)
                                doc_errors += 1
                                with log_file_lock:
                                    with open(log_file_path, "a") as log_file:
                                        log_file.write(f"    ERROR: {str(e)}\n")
                            
                            # Update count
                            doc_chunks_processed += 1
                            
                            # Update resume log with completed chunk ID (thread-safe)
                            with resume_log_lock:
                                with open(resume_log_path, "a") as f:
                                    f.write(f"{chunk['id']}\n")
                    
                    # Log document completion (thread-safe)
                    doc_end_time = datetime.now()
                    doc_duration = (doc_end_time - doc_start_time).total_seconds()
                    with log_file_lock:
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"Document {doc_num}/{len(docs_to_chunks)} completed at {doc_end_time}\n")
                            log_file.write(f"  Duration: {doc_duration:.2f}s\n")
                            log_file.write(f"  Events extracted: {doc_events}\n")
                            log_file.write(f"  Errors: {doc_errors}\n\n")
                    
                    # Log completion
                    console.log(f"[green]Document {doc_num} complete - Found {doc_events} events in {doc_duration:.1f}s")
                    
                    # Return statistics
                    return doc_events, doc_chunks_processed, doc_api_calls, doc_errors
                
                # Submit documents for parallel processing
                console.log(f"[bold green]Processing {len(docs_to_chunks)} documents in parallel with {max_workers} workers")
                
                # Adjust max_workers to be no more than the number of documents
                actual_workers = min(max_workers, len(docs_to_chunks))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    # Create a dictionary of futures to track document processing
                    future_to_doc = {
                        executor.submit(process_document, i+1, (doc_key, doc_chunks)): i 
                        for i, (doc_key, doc_chunks) in enumerate(docs_to_chunks.items())
                    }
                    
                    # Process documents as they complete
                    for future in concurrent.futures.as_completed(future_to_doc):
                        doc_idx = future_to_doc[future]
                        try:
                            # Get results from this document processing
                            doc_events, doc_processed, doc_api_calls, doc_errors = future.result()
                            
                            # Update shared statistics (thread-safe)
                            with stats_lock:
                                shared_stats["total_events"] += doc_events
                                shared_stats["total_processed"] += doc_processed
                                shared_stats["total_api_calls"] += doc_api_calls
                                shared_stats["total_api_errors"] += doc_errors
                            
                            # Update progress
                            progress.update(overall_task, advance=1)
                            
                        except Exception as e:
                            console.log(f"[bold red]Error processing document {doc_idx+1}: {str(e)}")
                            with stats_lock:
                                shared_stats["total_api_errors"] += 1
                
                # Update the main counters from shared statistics
                total_events = shared_stats["total_events"]
                total_processed = shared_stats["total_processed"]
                total_api_calls = shared_stats["total_api_calls"]
                total_api_errors = shared_stats["total_api_errors"]
                
            else:
                # Process each document's chunks with context sequentially
                doc_num = 0
                for doc_key, doc_chunks in docs_to_chunks.items():
                    doc_num += 1
                    
                    if not doc_chunks:
                        continue
                        
                    # Log document processing start
                    doc_name = doc_chunks[0].get("file_name", "Unknown")
                    doc_pages = max([c.get("page_num", 0) for c in doc_chunks])
                    console.log(f"[bold blue]Processing document {doc_num}/{len(docs_to_chunks)}: {doc_name} ({doc_pages} pages)")
                    
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Document {doc_num}/{len(docs_to_chunks)} - {doc_name} started at {datetime.now()}\n")
                        log_file.write(f"  Contains {len(doc_chunks)} chunks across {doc_pages} pages\n")
                    
                    # Process document chunks in small batches for manageable context
                    doc_batch_size = min(5, len(doc_chunks))  # Use smaller batches for document processing
                    doc_events = 0
                    doc_errors = 0
                    doc_start_time = datetime.now()
                    
                    # Process groups of chunks with document context
                    for i in range(0, len(doc_chunks), doc_batch_size):
                        context_batch = doc_chunks[i:i+doc_batch_size]
                        
                        for j, chunk in enumerate(context_batch):
                            chunk_start_time = datetime.now()
                            try:
                                # Log extract attempt
                                with open(log_file_path, "a") as log_file:
                                    log_file.write(f"  Chunk {j+1}/{len(context_batch)} (ID: {chunk['id']}) - Processing at {chunk_start_time}\n")
                                
                                # Add document context to the chunk for enhanced extraction
                                if len(context_batch) > 1:
                                    # Add surrounding context to the chunk
                                    chunk_with_context = chunk.copy()
                                    
                                    # Include information about surrounding chunks
                                    context_info = f"\nDocument context: This is chunk {j+1} of {len(context_batch)} from page {chunk.get('page_num', 'unknown')}.\n"
                                    
                                    # Add previous chunk summary if available
                                    if j > 0:
                                        prev_chunk = context_batch[j-1]
                                        context_info += f"\nPrevious chunk from page {prev_chunk.get('page_num', 'unknown')} begins with: {prev_chunk.get('text', '')[:150]}...\n"
                                    
                                    # Add next chunk summary if available
                                    if j < len(context_batch) - 1:
                                        next_chunk = context_batch[j+1]
                                        context_info += f"\nNext chunk from page {next_chunk.get('page_num', 'unknown')} begins with: {next_chunk.get('text', '')[:150]}...\n"
                                    
                                    # Append context to chunk text
                                    chunk_with_context["enhanced_context"] = context_info
                                else:
                                    chunk_with_context = chunk
                                
                                # Process chunk with context
                                total_api_calls += 1
                                chunk_events = extract_timeline_events(chunk_with_context, matter_id, batch_mode=True, document_context=True)
                                doc_events += len(chunk_events)
                                total_events += len(chunk_events)
                                
                                # Log result
                                chunk_end_time = datetime.now()
                                duration = (chunk_end_time - chunk_start_time).total_seconds()
                                with open(log_file_path, "a") as log_file:
                                    log_file.write(f"    Completed in {duration:.2f}s - Found {len(chunk_events)} events\n")
                                
                            except Exception as e:
                                # Log error
                                doc_errors += 1
                                total_api_errors += 1
                                with open(log_file_path, "a") as log_file:
                                    log_file.write(f"    ERROR: {str(e)}\n")
                            
                            # Update progress and count
                            total_processed += 1
                            
                            # Update resume log with completed chunk ID
                            with open(resume_log_path, "a") as f:
                                f.write(f"{chunk['id']}\n")
                    
                    # Log document completion
                    doc_end_time = datetime.now()
                    doc_duration = (doc_end_time - doc_start_time).total_seconds()
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Document {doc_num}/{len(docs_to_chunks)} completed at {doc_end_time}\n")
                        log_file.write(f"  Duration: {doc_duration:.2f}s\n")
                        log_file.write(f"  Events extracted: {doc_events}\n")
                        log_file.write(f"  Errors: {doc_errors}\n\n")
                    
                    console.log(f"[green]Document {doc_num} complete - Found {doc_events} events in {doc_duration:.1f}s")
                    
                    # Update progress for document
                    progress.update(overall_task, advance=1)
                
        else:
            # Original batch-based processing without document awareness
            overall_task = progress.add_task(f"Processing {len(chunks)} chunks", total=len(chunks))
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                # Log batch start
                console.log(f"[bold blue]Starting batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Batch {batch_num}/{total_batches} started at {datetime.now()}\n")
                
                # Update progress description
                progress.update(
                    overall_task,
                    advance=0,
                    description=f"Batch {batch_num}/{total_batches}"
                )
                
                # Process batch with detailed logging
                batch_start_time = datetime.now()
                events = []
                
                # Process each chunk individually to log progress
                batch_events = 0
                batch_errors = 0
                
                for j, chunk in enumerate(batch):
                    chunk_start_time = datetime.now()
                    try:
                        # Log extract attempt
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"  Chunk {j+1}/{len(batch)} (ID: {chunk['id']}) - Processing at {chunk_start_time}\n")
                        
                        # Process chunk
                        total_api_calls += 1
                        chunk_events = extract_timeline_events(chunk, matter_id, batch_mode=True)
                        events.extend(chunk_events)
                        batch_events += len(chunk_events)
                        
                        # Log result
                        chunk_end_time = datetime.now()
                        duration = (chunk_end_time - chunk_start_time).total_seconds()
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"    Completed in {duration:.2f}s - Found {len(chunk_events)} events\n")
                        
                    except Exception as e:
                        # Log error
                        batch_errors += 1
                        total_api_errors += 1
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"    ERROR: {str(e)}\n")
                    
                    # Update progress for individual chunk
                    progress.update(overall_task, advance=1/len(batch))
                    total_processed += 1
                
                # Update resume log with completed chunk IDs
                with open(resume_log_path, "a") as f:
                    for chunk in batch:
                        f.write(f"{chunk['id']}\n")
                
                # Add batch results to total
                total_events += batch_events
            
            # Log batch completion
            batch_end_time = datetime.now()
            batch_duration = (batch_end_time - batch_start_time).total_seconds()
            console.log(f"[bold green]Batch {batch_num} complete in {batch_duration:.2f}s - Found {batch_events} events")
            
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Batch {batch_num}/{total_batches} completed at {batch_end_time}\n")
                log_file.write(f"  Duration: {batch_duration:.2f}s\n")
                log_file.write(f"  Events extracted: {batch_events}\n")
                log_file.write(f"  Errors: {batch_errors}\n\n")
            
            # Update overall progress
            progress.update(overall_task, advance=len(batch) - 1)
    
    finally:
        # Stop progress bar if we created it
        if progress_owner:
            progress.stop()
        
        # Write final summary to log
        with open(log_file_path, "a") as log_file:
            log_file.write(f"===============================================\n")
            log_file.write(f"Timeline extraction completed at {datetime.now()}\n")
            log_file.write(f"Total chunks processed: {total_processed}/{len(chunks)}\n")
            log_file.write(f"Total events extracted: {total_events}\n")
            log_file.write(f"Total API calls: {total_api_calls}\n")
            log_file.write(f"Total API errors: {total_api_errors}\n")
            log_file.write(f"===============================================\n\n")
        
        # Print summary to console
        console.log(f"[bold green]Timeline extraction complete")
        console.log(f"Processed {total_processed} chunks, found {total_events} events")
        if total_api_errors > 0:
            console.log(f"[bold red]Encountered {total_api_errors} API errors - check log for details")
        console.log(f"[dim]Detailed log saved to {log_file_path}")
    
    return total_events