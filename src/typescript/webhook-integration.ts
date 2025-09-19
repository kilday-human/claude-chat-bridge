/**
 * TypeScript Webhook Integration for GPT-Bridge
 * Production-ready webhook system with reliability, monitoring, and security
 */

// Type definitions matching Python implementation
export interface QueryData {
  summary: string; // First 50 chars
  category: 'code_review' | 'debugging' | 'architecture' | 'general';
  estimated_complexity: 1 | 2 | 3;
}

export interface ResponseData {
  model_used: 'gpt-4' | 'claude-3';
  response_time_ms: number;
  estimated_tokens: number;
  success: boolean;
}

export interface UsagePayload {
  timestamp: string;
  session_id: string;
  query: QueryData;
  response: ResponseData;
  user_agent?: string;
}

export interface WebhookAttempt {
  attempt_number: number;
  timestamp: string;
  status: 'pending' | 'success' | 'failed' | 'dead_letter';
  error_message?: string;
  response_code?: number;
  response_time_ms?: number;
}

export interface WebhookMessage {
  id: string;
  payload: UsagePayload;
  webhook_url: string;
  created_at: string;
  attempts: WebhookAttempt[];
  max_retries: number;
  retry_delay_seconds: number;
  status: 'pending' | 'success' | 'failed' | 'dead_letter';
}

export interface WebhookConfig {
  webhook_url: string;
  enabled: boolean;
  max_retries: number;
  timeout_seconds: number;
  retry_delay_seconds: number;
  webhook_secret?: string;
  signature_header: string;
  rate_limit_per_hour: number;
}

export interface WebhookMetrics {
  total_sent: number;
  successful: number;
  failed: number;
  dead_letter: number;
  avg_response_time_ms: number;
  last_success?: string;
  last_failure?: string;
  success_rate: number;
  failure_rate: number;
}

export interface Alert {
  id: string;
  level: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: string;
  resolved: boolean;
  resolved_at?: string;
}

// Webhook Reliability System
export class WebhookRetryManager {
  private maxRetries: number;
  private baseDelay: number;
  private maxDelay: number;
  private deadLetterQueue: WebhookMessage[] = [];

  constructor(maxRetries: number = 3, baseDelay: number = 1000, maxDelay: number = 300000) {
    this.maxRetries = maxRetries;
    this.baseDelay = baseDelay;
    this.maxDelay = maxDelay;
  }

  private calculateDelay(attemptNumber: number): number {
    return Math.min(this.baseDelay * Math.pow(2, attemptNumber - 1), this.maxDelay);
  }

  async sendWithRetry(message: WebhookMessage): Promise<boolean> {
    for (let attemptNum = 1; attemptNum <= this.maxRetries; attemptNum++) {
      try {
        // Calculate delay for this attempt
        if (attemptNum > 1) {
          const delay = this.calculateDelay(attemptNum);
          console.log(`Retrying webhook ${message.id} in ${delay}ms (attempt ${attemptNum})`);
          await this.sleep(delay);
        }

        // Attempt to send webhook
        const startTime = Date.now();
        const success = await this.sendWebhook(message);
        const responseTime = Date.now() - startTime;

        // Record attempt
        const attempt: WebhookAttempt = {
          attempt_number: attemptNum,
          timestamp: new Date().toISOString(),
          status: success ? 'success' : 'failed',
          response_time_ms: responseTime
        };
        message.attempts.push(attempt);
        message.status = attempt.status;

        if (success) {
          console.log(`Webhook ${message.id} sent successfully on attempt ${attemptNum}`);
          return true;
        } else {
          console.warn(`Webhook ${message.id} failed on attempt ${attemptNum}`);
        }
      } catch (error) {
        console.error(`Unexpected error sending webhook ${message.id}:`, error);
        const attempt: WebhookAttempt = {
          attempt_number: attemptNum,
          timestamp: new Date().toISOString(),
          status: 'failed',
          error_message: error instanceof Error ? error.message : String(error)
        };
        message.attempts.push(attempt);
        message.status = attempt.status;
      }
    }

    // All retries failed, move to dead letter queue
    message.status = 'dead_letter';
    this.deadLetterQueue.push(message);
    console.warn(`Message ${message.id} moved to dead letter queue after ${this.maxRetries} attempts`);
    return false;
  }

  private async sendWebhook(message: WebhookMessage): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const response = await fetch(message.webhook_url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(message.payload),
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Timeout after 10 seconds');
      }
      throw error;
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getDeadLetterQueue(): WebhookMessage[] {
    return [...this.deadLetterQueue];
  }
}

// Session Persistence System
export class SessionPersistenceManager {
  private sessions: Map<string, any> = new Map();
  private deliveryCache: Set<string> = new Set();

  generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  getOrCreateSession(sessionId: string): any {
    if (!this.sessions.has(sessionId)) {
      this.sessions.set(sessionId, {
        session_id: sessionId,
        created_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
        query_count: 0,
        total_tokens: 0,
        total_cost: 0,
        models_used: new Set(),
        categories: new Set(),
        metadata: {}
      });
    }
    return this.sessions.get(sessionId);
  }

  updateSession(sessionId: string, updates: Partial<any>): void {
    const session = this.getOrCreateSession(sessionId);
    Object.assign(session, updates);
    session.last_activity = new Date().toISOString();
  }

  isDuplicateWebhook(payload: UsagePayload): boolean {
    const payloadStr = JSON.stringify(payload, Object.keys(payload).sort());
    const payloadHash = this.hashString(payloadStr);
    
    if (this.deliveryCache.has(payloadHash)) {
      console.log(`Duplicate webhook detected: ${payloadHash.substring(0, 16)}...`);
      return true;
    }
    
    this.deliveryCache.add(payloadHash);
    return false;
  }

  private hashString(str: string): string {
    // Simple hash function for demo - in production use crypto.subtle.digest
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(16);
  }
}

// Monitoring and Metrics System
export class WebhookMetricsCollector {
  private metrics: Map<string, number> = new Map();
  private responseTimes: number[] = [];

  incrementCounter(name: string, value: number = 1): void {
    this.metrics.set(name, (this.metrics.get(name) || 0) + value);
  }

  setGauge(name: string, value: number): void {
    this.metrics.set(name, value);
  }

  recordTiming(name: string, durationMs: number): void {
    this.responseTimes.push(durationMs);
    if (this.responseTimes.length > 1000) {
      this.responseTimes = this.responseTimes.slice(-1000); // Keep last 1000
    }
    this.incrementCounter(`${name}_count`);
  }

  getCounter(name: string): number {
    return this.metrics.get(name) || 0;
  }

  getGauge(name: string): number {
    return this.metrics.get(name) || 0;
  }

  getAverageResponseTime(): number {
    if (this.responseTimes.length === 0) return 0;
    return this.responseTimes.reduce((a, b) => a + b, 0) / this.responseTimes.length;
  }

  getMetrics(): WebhookMetrics {
    const totalSent = this.getCounter('webhook_total');
    const successful = this.getCounter('webhook_success');
    const failed = this.getCounter('webhook_failure');
    const successRate = totalSent > 0 ? successful / totalSent : 0;

    return {
      total_sent: totalSent,
      successful,
      failed,
      dead_letter: this.getCounter('webhook_dead_letter'),
      avg_response_time_ms: this.getAverageResponseTime(),
      last_success: this.metrics.get('last_success')?.toString(),
      last_failure: this.metrics.get('last_failure')?.toString(),
      success_rate: successRate,
      failure_rate: 1 - successRate
    };
  }
}

// Security System
export class WebhookSecurityManager {
  private config: WebhookConfig;
  private rateLimiter: Map<string, { count: number; resetTime: number }> = new Map();

  constructor(config: WebhookConfig) {
    this.config = config;
  }

  verifyRequest(payload: string, headers: Record<string, string>, clientIp?: string): boolean {
    // Check rate limiting
    if (!this.isRateLimitAllowed()) {
      throw new Error('Rate limit exceeded');
    }

    // Verify signature if secret is provided
    if (this.config.webhook_secret) {
      const signature = headers[this.config.signature_header];
      if (!signature || !this.verifySignature(payload, signature)) {
        throw new Error('Invalid webhook signature');
      }
    }

    return true;
  }

  private isRateLimitAllowed(): boolean {
    const now = Date.now();
    const windowStart = now - (60 * 60 * 1000); // 1 hour window
    const key = 'webhook_calls';
    
    const rateLimit = this.rateLimiter.get(key);
    if (!rateLimit || rateLimit.resetTime < now) {
      this.rateLimiter.set(key, { count: 1, resetTime: now + (60 * 60 * 1000) });
      return true;
    }

    if (rateLimit.count >= this.config.rate_limit_per_hour) {
      return false;
    }

    rateLimit.count++;
    return true;
  }

  private verifySignature(payload: string, signature: string): boolean {
    if (!this.config.webhook_secret) return true;

    // In a real implementation, you'd use crypto.subtle.digest
    // For demo purposes, using a simple HMAC-like approach
    const expectedSignature = this.generateSignature(payload);
    return signature === expectedSignature;
  }

  generateSignature(payload: string): string {
    if (!this.config.webhook_secret) return '';
    
    // In production, use proper HMAC-SHA256
    // This is a simplified version for demo
    const combined = this.config.webhook_secret + payload;
    return 'sha256=' + btoa(combined).replace(/[^a-zA-Z0-9]/g, '').substring(0, 64);
  }
}

// Main Production Webhook Client
export class ProductionWebhookClient {
  private config: WebhookConfig;
  private retryManager: WebhookRetryManager;
  private sessionManager: SessionPersistenceManager;
  private metricsCollector: WebhookMetricsCollector;
  private securityManager: WebhookSecurityManager;
  private endpointValidated: boolean = false;

  constructor(config: WebhookConfig) {
    this.config = config;
    this.retryManager = new WebhookRetryManager(config.max_retries);
    this.sessionManager = new SessionPersistenceManager();
    this.metricsCollector = new WebhookMetricsCollector();
    this.securityManager = new WebhookSecurityManager(config);
  }

  async validateEndpoint(): Promise<boolean> {
    try {
      const testPayload = {
        test: true,
        timestamp: new Date().toISOString(),
        message: 'Health check from GPT-Bridge'
      };

      const response = await fetch(this.config.webhook_url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testPayload)
      });

      this.endpointValidated = response.ok;
      
      if (this.endpointValidated) {
        console.log('Webhook endpoint validated successfully');
      } else {
        console.error(`Webhook endpoint validation failed: HTTP ${response.status}`);
      }

      return this.endpointValidated;
    } catch (error) {
      console.error('Webhook endpoint validation error:', error);
      this.endpointValidated = false;
      return false;
    }
  }

  async sendWebhook(payload: UsagePayload): Promise<boolean> {
    if (!this.config.enabled) {
      console.log('Webhook disabled, skipping');
      return false;
    }

    if (!this.endpointValidated) {
      console.warn('Webhook endpoint not validated, attempting validation...');
      if (!await this.validateEndpoint()) {
        console.error('Cannot send webhook: endpoint validation failed');
        return false;
      }
    }

    // Check for duplicates
    if (this.sessionManager.isDuplicateWebhook(payload)) {
      console.log('Duplicate webhook detected, skipping');
      return true;
    }

    // Create webhook message
    const messageId = this.generateMessageId(payload);
    const message: WebhookMessage = {
      id: messageId,
      payload,
      webhook_url: this.config.webhook_url,
      created_at: new Date().toISOString(),
      attempts: [],
      max_retries: this.config.max_retries,
      retry_delay_seconds: this.config.retry_delay_seconds,
      status: 'pending'
    };

    // Record metrics
    this.metricsCollector.incrementCounter('webhook_total');

    // Send with retry logic
    const success = await this.retryManager.sendWithRetry(message);

    // Update metrics
    if (success) {
      this.metricsCollector.incrementCounter('webhook_success');
      this.metricsCollector.setGauge('last_success', Date.now());
    } else {
      this.metricsCollector.incrementCounter('webhook_failure');
      this.metricsCollector.incrementCounter('webhook_dead_letter');
      this.metricsCollector.setGauge('last_failure', Date.now());
    }

    return success;
  }

  private generateMessageId(payload: UsagePayload): string {
    const payloadStr = JSON.stringify(payload) + Date.now();
    return btoa(payloadStr).replace(/[^a-zA-Z0-9]/g, '').substring(0, 16);
  }

  getHealthStatus(): any {
    const metrics = this.metricsCollector.getMetrics();
    return {
      endpoint_validated: this.endpointValidated,
      metrics,
      dead_letter_count: this.retryManager.getDeadLetterQueue().length,
      config: {
        enabled: this.config.enabled,
        max_retries: this.config.max_retries,
        timeout_seconds: this.config.timeout_seconds
      }
    };
  }

  getMetrics(): WebhookMetrics {
    return this.metricsCollector.getMetrics();
  }
}

// Utility functions
export class WebhookUtils {
  static categorizeQuery(query: string): QueryData['category'] {
    const queryLower = query.toLowerCase();
    
    if (['review', 'code review', 'pull request', 'pr', 'refactor'].some(word => queryLower.includes(word))) {
      return 'code_review';
    }
    
    if (['debug', 'error', 'bug', 'fix', 'issue', 'problem'].some(word => queryLower.includes(word))) {
      return 'debugging';
    }
    
    if (['architecture', 'design', 'structure', 'system', 'pattern'].some(word => queryLower.includes(word))) {
      return 'architecture';
    }
    
    return 'general';
  }

  static estimateComplexity(query: string, responseTimeMs: number): QueryData['estimated_complexity'] {
    const queryLength = query.length;
    
    if (queryLength > 200 || responseTimeMs > 3000) {
      return 3; // High complexity
    } else if (queryLength > 100 || responseTimeMs > 1000) {
      return 2; // Medium complexity
    } else {
      return 1; // Low complexity
    }
  }

  static determineModelType(modelName: string): ResponseData['model_used'] {
    if (modelName.toLowerCase().includes('gpt')) {
      return 'gpt-4';
    } else if (modelName.toLowerCase().includes('claude')) {
      return 'claude-3';
    } else {
      return 'gpt-4'; // Default fallback
    }
  }
}

// Export default configuration
export const defaultWebhookConfig: WebhookConfig = {
  webhook_url: process.env.ZAPIER_WEBHOOK_URL || '',
  enabled: process.env.ENABLE_WEBHOOK === 'true',
  max_retries: parseInt(process.env.WEBHOOK_MAX_RETRIES || '3'),
  timeout_seconds: parseInt(process.env.WEBHOOK_TIMEOUT_SECONDS || '10'),
  retry_delay_seconds: parseInt(process.env.WEBHOOK_RETRY_DELAY_SECONDS || '60'),
  webhook_secret: process.env.WEBHOOK_SECRET,
  signature_header: process.env.WEBHOOK_SIGNATURE_HEADER || 'X-Webhook-Signature',
  rate_limit_per_hour: parseInt(process.env.WEBHOOK_RATE_LIMIT || '100')
};
