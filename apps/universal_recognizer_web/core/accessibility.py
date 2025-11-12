"""
Accessibility detection and suggestions for universal character recognition.
"""

from typing import Dict, List, Optional


# Common dyslexia confusion pairs
DYSLEXIA_CONFUSION_PAIRS = {
    'b': 'd',
    'd': 'b',
    'p': 'q',
    'q': 'p',
    'n': 'u',
    'u': 'n',
    'm': 'w',
    'w': 'm',
    '6': '9',
    '9': '6',
    'I': 'l',
    'l': 'I',
    'O': '0',
    '0': 'O'
}


def detect_potential_issues(prediction_result: Dict, mirror_result: Optional[Dict] = None, 
                           quality_metrics: Optional[Dict] = None) -> List[Dict]:
    """
    Detect potential accessibility issues from prediction results.
    
    Args:
        prediction_result: Standard prediction result
        mirror_result: Mirror detection result (if available)
        quality_metrics: Writing quality metrics (if available)
    
    Returns:
        List of detected issues with severity and type
    """
    issues = []
    
    if prediction_result is None:
        return issues
    
    predicted_char = prediction_result.get('predicted_character', '')
    confidence = prediction_result.get('confidence', 0.0)
    
    # Low confidence issue
    if confidence < 60.0:
        issues.append({
            'type': 'low_confidence',
            'severity': 'medium',
            'message': f'Low confidence ({confidence:.1f}%) - try drawing more clearly',
            'suggestion': 'Make your strokes bolder and keep the character centered'
        })
    elif confidence < 40.0:
        issues.append({
            'type': 'very_low_confidence',
            'severity': 'high',
            'message': f'Very low confidence ({confidence:.1f}%) - character may be unclear',
            'suggestion': 'Try redrawing the character with clearer, bolder strokes'
        })
    
    # Mirror detection issue
    if mirror_result and mirror_result.get('mirror_detected', False):
        mirrored_char = mirror_result.get('predicted_character', '')
        issues.append({
            'type': 'mirror_detected',
            'severity': 'high',
            'message': f'Mirror detected: Did you mean to write "{mirrored_char}"?',
            'suggestion': f'Your character appears mirrored. Try writing "{mirrored_char}" in the correct orientation.',
            'original_char': predicted_char,
            'mirrored_char': mirrored_char
        })
    
    # Dyslexia confusion pattern detection
    if predicted_char in DYSLEXIA_CONFUSION_PAIRS:
        confused_char = DYSLEXIA_CONFUSION_PAIRS[predicted_char]
        
        # Check if confused character is in top predictions
        top_predictions = prediction_result.get('top_predictions', [])
        confused_in_top = any(
            pred.get('character') == confused_char 
            for pred in top_predictions[:3]  # Check top 3
        )
        
        if confused_in_top:
            issues.append({
                'type': 'dyslexia_confusion',
                'severity': 'medium',
                'message': f'Common confusion: "{predicted_char}" and "{confused_char}" look similar',
                'suggestion': f'Make sure you wrote "{predicted_char}" and not "{confused_char}". Pay attention to the direction.',
                'confused_pair': (predicted_char, confused_char)
            })
    
    # Writing quality issues
    if quality_metrics:
        overall_score = quality_metrics.get('overall_score', 100.0)
        clarity_score = quality_metrics.get('clarity_score', 100.0)
        centering_score = quality_metrics.get('centering_score', 100.0)
        size_score = quality_metrics.get('size_score', 100.0)
        
        if overall_score < 50.0:
            issues.append({
                'type': 'poor_quality',
                'severity': 'medium',
                'message': 'Writing quality could be improved',
                'suggestion': 'Try making your strokes clearer and keeping the character well-centered'
            })
        
        if clarity_score < 40.0:
            issues.append({
                'type': 'low_clarity',
                'severity': 'low',
                'message': 'Strokes could be bolder',
                'suggestion': 'Press harder or use a thicker brush to make your strokes more visible'
            })
        
        if centering_score < 50.0:
            issues.append({
                'type': 'poor_centering',
                'severity': 'low',
                'message': 'Character could be better centered',
                'suggestion': 'Try to draw the character in the center of the canvas'
            })
        
        if size_score < 30.0:
            issues.append({
                'type': 'too_small',
                'severity': 'low',
                'message': 'Character is quite small',
                'suggestion': 'Try drawing a larger character that fills more of the canvas'
            })
        elif size_score > 90.0:
            issues.append({
                'type': 'too_large',
                'severity': 'low',
                'message': 'Character is very large',
                'suggestion': 'Try drawing a slightly smaller character so it fits better'
            })
    
    return issues


def generate_suggestions(issues: List[Dict], predicted_char: str) -> List[Dict]:
    """
    Generate helpful suggestions based on detected issues.
    
    Args:
        issues: List of detected issues
        predicted_char: The predicted character
    
    Returns:
        List of suggestions with actionable advice
    """
    suggestions = []
    
    # Sort issues by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_issues = sorted(issues, key=lambda x: severity_order.get(x.get('severity', 'low'), 2))
    
    for issue in sorted_issues:
        suggestion = {
            'type': issue.get('type'),
            'priority': issue.get('severity', 'low'),
            'message': issue.get('message', ''),
            'advice': issue.get('suggestion', ''),
            'category': _categorize_issue(issue.get('type'))
        }
        suggestions.append(suggestion)
    
    # Add character-specific guidance if no major issues
    if len(issues) == 0 or all(i.get('severity') == 'low' for i in issues):
        char_guidance = _get_character_guidance(predicted_char)
        if char_guidance:
            suggestions.append({
                'type': 'character_guidance',
                'priority': 'info',
                'message': f'Writing tip for "{predicted_char}"',
                'advice': char_guidance,
                'category': 'guidance'
            })
    
    return suggestions


def _categorize_issue(issue_type: str) -> str:
    """Categorize issue type."""
    if issue_type in ['mirror_detected', 'dyslexia_confusion']:
        return 'accessibility'
    elif issue_type in ['low_confidence', 'very_low_confidence', 'poor_quality']:
        return 'quality'
    elif issue_type in ['low_clarity', 'poor_centering', 'too_small', 'too_large']:
        return 'technique'
    else:
        return 'general'


def _get_character_guidance(char: str) -> Optional[str]:
    """Get character-specific writing guidance."""
    guidance = {
        'b': 'Start from the top, draw a straight line down, then add the curve on the right',
        'd': 'Start with the curve on the left, then add the straight line',
        'p': 'Draw the line down first, then add the curve at the bottom',
        'q': 'Draw the curve first, then add the line going down',
        '6': 'Start from the top, curve down and around',
        '9': 'Start from the bottom, curve up and around',
        '0': 'Draw a complete circle or oval',
        'O': 'Similar to zero, but make sure it\'s clearly a letter',
        'I': 'A single vertical line - make it straight',
        'l': 'Similar to I, but make sure it\'s lowercase',
        'a': 'Start with the circle, then add the line on the right',
        'e': 'Start with the horizontal line, then add the curve',
        's': 'Draw an S-curve, making sure both curves are clear'
    }
    return guidance.get(char.lower())


def get_resources(issue_type: str) -> List[Dict]:
    """
    Get educational resources based on issue type.
    
    Args:
        issue_type: Type of issue detected
    
    Returns:
        List of resources with links and descriptions
    """
    resources = []
    
    if issue_type in ['mirror_detected', 'dyslexia_confusion']:
        resources.extend([
            {
                'title': 'Understanding Letter Reversals',
                'description': 'Learn about why letter reversals happen and how to practice',
                'category': 'dyslexia',
                'url': 'https://www.understood.org/en/articles/why-kids-reverse-letters'
            },
            {
                'title': 'Handwriting Practice for Kids',
                'description': 'Free printable worksheets for practicing letter formation',
                'category': 'practice',
                'url': 'https://www.education.com/worksheets/handwriting/'
            }
        ])
    
    if issue_type in ['low_confidence', 'poor_quality']:
        resources.extend([
            {
                'title': 'Improving Handwriting Skills',
                'description': 'Tips and exercises for clearer handwriting',
                'category': 'technique',
                'url': 'https://www.handwritingforkids.com/'
            }
        ])
    
    # General resources
    resources.extend([
        {
            'title': 'NeuralEngine Documentation',
            'description': 'Learn more about how character recognition works',
            'category': 'general',
            'url': '#'
        }
    ])
    
    return resources


def format_accessibility_report(prediction_result: Dict, mirror_result: Optional[Dict] = None,
                               quality_metrics: Optional[Dict] = None) -> Dict:
    """
    Create a comprehensive accessibility report.
    
    Args:
        prediction_result: Standard prediction result
        mirror_result: Mirror detection result
        quality_metrics: Writing quality metrics
    
    Returns:
        Complete accessibility report
    """
    issues = detect_potential_issues(prediction_result, mirror_result, quality_metrics)
    suggestions = generate_suggestions(issues, prediction_result.get('predicted_character', ''))
    
    # Get resources for all issue types
    all_resources = []
    for issue in issues:
        resources = get_resources(issue.get('type', ''))
        all_resources.extend(resources)
    
    # Remove duplicates
    seen_urls = set()
    unique_resources = []
    for resource in all_resources:
        if resource['url'] not in seen_urls:
            seen_urls.add(resource['url'])
            unique_resources.append(resource)
    
    return {
        'issues': issues,
        'suggestions': suggestions,
        'resources': unique_resources,
        'has_issues': len(issues) > 0,
        'severity_summary': {
            'high': sum(1 for i in issues if i.get('severity') == 'high'),
            'medium': sum(1 for i in issues if i.get('severity') == 'medium'),
            'low': sum(1 for i in issues if i.get('severity') == 'low')
        }
    }

