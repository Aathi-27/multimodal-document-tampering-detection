"""Alerting and notification integration module.

Sends alerts for HIGH-risk document detections via email, Slack, and SMS.
All integrations use environment variables for configuration.

Configuration (set as environment variables or in .env file):
    # Email (AWS SES)
    ALERT_EMAIL_ENABLED=true
    ALERT_EMAIL_FROM=fraud-alerts@yourcompany.com
    ALERT_EMAIL_TO=security-team@yourcompany.com
    AWS_SES_REGION=us-east-1
    
    # Slack
    ALERT_SLACK_ENABLED=true
    ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
    
    # SMS (Twilio)
    ALERT_SMS_ENABLED=true
    ALERT_TWILIO_SID=your_account_sid
    ALERT_TWILIO_TOKEN=your_auth_token
    ALERT_TWILIO_FROM=+1234567890
    ALERT_TWILIO_TO=+0987654321
    
    # Webhook (generic)
    ALERT_WEBHOOK_ENABLED=true
    ALERT_WEBHOOK_URL=https://your-api.com/webhooks/fraud-alert

Usage:
    from alerting import AlertManager
    alerts = AlertManager()
    alerts.send_alert(risk_tier="HIGH", risk_score=0.85, document_name="paystub_123.jpg",
                      explanations=["Visual tamper overlaps OCR text regions"])
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage fraud detection alerts across multiple channels.
    
    Supports:
    - Email via AWS SES
    - Slack webhook notifications
    - SMS via Twilio
    - Generic webhook (for Jira, ServiceNow, etc.)
    
    All channels are optional and configured via environment variables.
    Failed notifications are logged but don't block the pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize alert manager.
        
        Args:
            config: Optional config dict. If None, reads from environment variables.
        """
        if config is None:
            config = self._load_env_config()
        self.config = config
        
        self.email_enabled = config.get("ALERT_EMAIL_ENABLED", "false").lower() == "true"
        self.slack_enabled = config.get("ALERT_SLACK_ENABLED", "false").lower() == "true"
        self.sms_enabled = config.get("ALERT_SMS_ENABLED", "false").lower() == "true"
        self.webhook_enabled = config.get("ALERT_WEBHOOK_ENABLED", "false").lower() == "true"
        
        # Alert threshold: only alert for risk tiers at or above this level
        self.alert_tiers = config.get("ALERT_TIERS", "HIGH").split(",")
    
    def _load_env_config(self) -> Dict[str, str]:
        """Load configuration from environment variables."""
        return {key: os.environ.get(key, "") for key in [
            "ALERT_EMAIL_ENABLED", "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO", "AWS_SES_REGION",
            "ALERT_SLACK_ENABLED", "ALERT_SLACK_WEBHOOK",
            "ALERT_SMS_ENABLED", "ALERT_TWILIO_SID", "ALERT_TWILIO_TOKEN",
            "ALERT_TWILIO_FROM", "ALERT_TWILIO_TO",
            "ALERT_WEBHOOK_ENABLED", "ALERT_WEBHOOK_URL",
            "ALERT_TIERS",
        ]}
    
    def send_alert(self, risk_tier: str, risk_score: float,
                   document_name: str = "Unknown",
                   explanations: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """Send alert if risk tier meets threshold.
        
        Args:
            risk_tier: Risk classification ("LOW", "MEDIUM", "HIGH").
            risk_score: Continuous risk score (0-1).
            document_name: Document filename or identifier.
            explanations: List of human-readable explanation strings.
            metadata: Additional metadata dict (scores, timings, etc.).
            
        Returns:
            Dict mapping channel name to success boolean.
        """
        results = {}
        
        if risk_tier not in self.alert_tiers:
            logger.info(f"Risk tier {risk_tier} below alert threshold {self.alert_tiers}, skipping alerts.")
            return results
        
        alert_payload = self._build_alert_payload(risk_tier, risk_score, document_name,
                                                   explanations, metadata)
        
        if self.email_enabled:
            results["email"] = self._send_email(alert_payload)
        
        if self.slack_enabled:
            results["slack"] = self._send_slack(alert_payload)
        
        if self.sms_enabled:
            results["sms"] = self._send_sms(alert_payload)
        
        if self.webhook_enabled:
            results["webhook"] = self._send_webhook(alert_payload)
        
        return results
    
    def _build_alert_payload(self, risk_tier: str, risk_score: float,
                              document_name: str, explanations: Optional[List[str]],
                              metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build standardized alert payload."""
        return {
            "timestamp": datetime.now().isoformat(),
            "alert_type": "document_tampering_detected",
            "risk_tier": risk_tier,
            "risk_score": risk_score,
            "document_name": document_name,
            "explanations": explanations or [],
            "metadata": metadata or {},
            "severity": "critical" if risk_tier == "HIGH" else "warning",
        }
    
    def _send_email(self, payload: Dict[str, Any]) -> bool:
        """Send alert email via AWS SES."""
        try:
            import boto3
            
            ses = boto3.client(
                "ses",
                region_name=self.config.get("AWS_SES_REGION", "us-east-1")
            )
            
            subject = f"🚨 [{payload['risk_tier']} RISK] Document Tampering Detected: {payload['document_name']}"
            
            body = f"""
Document Tampering Detection Alert
{'=' * 50}

Risk Tier: {payload['risk_tier']}
Risk Score: {payload['risk_score']:.3f}
Document: {payload['document_name']}
Timestamp: {payload['timestamp']}

Explanations:
{chr(10).join(f'  • {exp}' for exp in payload['explanations'])}

{'=' * 50}
This is an automated alert from the Document Tampering Detection System.
Please review the flagged document at your earliest convenience.
"""
            
            ses.send_email(
                Source=self.config.get("ALERT_EMAIL_FROM", ""),
                Destination={"ToAddresses": [self.config.get("ALERT_EMAIL_TO", "")]},
                Message={
                    "Subject": {"Data": subject},
                    "Body": {"Text": {"Data": body}},
                },
            )
            
            logger.info("Email alert sent successfully.")
            return True
            
        except ImportError:
            logger.warning("boto3 not installed. Email alerts unavailable. Install: pip install boto3")
            return False
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_slack(self, payload: Dict[str, Any]) -> bool:
        """Send alert to Slack via webhook."""
        try:
            import urllib.request
            
            webhook_url = self.config.get("ALERT_SLACK_WEBHOOK", "")
            if not webhook_url:
                logger.warning("Slack webhook URL not configured.")
                return False
            
            # Color based on risk tier
            colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}
            
            slack_message = {
                "attachments": [{
                    "color": colors.get(payload["risk_tier"], "#95a5a6"),
                    "title": f"🔍 Document Tampering Alert: {payload['document_name']}",
                    "fields": [
                        {"title": "Risk Tier", "value": payload["risk_tier"], "short": True},
                        {"title": "Risk Score", "value": f"{payload['risk_score']:.3f}", "short": True},
                        {"title": "Timestamp", "value": payload["timestamp"], "short": True},
                        {"title": "Explanations", "value": "\n".join(f"• {e}" for e in payload["explanations"][:5]) or "None", "short": False},
                    ],
                    "footer": "Document Tampering Detection System",
                }]
            }
            
            data = json.dumps(slack_message).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            
            logger.info("Slack alert sent successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_sms(self, payload: Dict[str, Any]) -> bool:
        """Send alert SMS via Twilio."""
        try:
            from twilio.rest import Client
            
            client = Client(
                self.config.get("ALERT_TWILIO_SID", ""),
                self.config.get("ALERT_TWILIO_TOKEN", ""),
            )
            
            message_body = (
                f"FRAUD ALERT: {payload['risk_tier']} risk detected. "
                f"Doc: {payload['document_name']}. "
                f"Score: {payload['risk_score']:.2f}. "
                f"Review required."
            )
            
            client.messages.create(
                body=message_body,
                from_=self.config.get("ALERT_TWILIO_FROM", ""),
                to=self.config.get("ALERT_TWILIO_TO", ""),
            )
            
            logger.info("SMS alert sent successfully.")
            return True
            
        except ImportError:
            logger.warning("twilio not installed. SMS alerts unavailable. Install: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            return False
    
    def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """Send alert to generic webhook endpoint."""
        try:
            import urllib.request
            
            webhook_url = self.config.get("ALERT_WEBHOOK_URL", "")
            if not webhook_url:
                logger.warning("Webhook URL not configured.")
                return False
            
            data = json.dumps(payload, default=str).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            
            logger.info("Webhook alert sent successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current alerting configuration status."""
        return {
            "email_enabled": self.email_enabled,
            "slack_enabled": self.slack_enabled,
            "sms_enabled": self.sms_enabled,
            "webhook_enabled": self.webhook_enabled,
            "alert_tiers": self.alert_tiers,
        }
