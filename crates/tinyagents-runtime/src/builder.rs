use std::sync::Arc;

use tinyagents_session::transcript::{TranscriptLocator, TranscriptMeta};

use crate::{
    NoopSessionHooks, PrefixSnapshot, RuntimeError, Session, SessionDriver, SessionHooks,
    ToolSnapshot, TranscriptCodec,
};

/// Configures a directly-owned, reusable [`Session`].
pub struct SessionBuilder<C: Clone + Send + Sync + 'static = ()> {
    driver: Arc<dyn SessionDriver<C>>,
    codec: Option<Arc<dyn TranscriptCodec<C>>>,
    hooks: Arc<dyn SessionHooks<C>>,
    prefix: PrefixSnapshot,
    tools: ToolSnapshot,
    transcript: Option<TranscriptConfig>,
}

struct TranscriptConfig {
    locator: Arc<dyn TranscriptLocator>,
    stem: String,
    meta: TranscriptMeta,
}

impl<C: Clone + Send + Sync + 'static> SessionBuilder<C> {
    /// Starts a builder over an object-safe execution driver.
    pub fn new(driver: Arc<dyn SessionDriver<C>>) -> Self {
        Self {
            driver,
            codec: None,
            hooks: Arc::new(NoopSessionHooks),
            prefix: PrefixSnapshot::default(),
            tools: ToolSnapshot::default(),
            transcript: None,
        }
    }

    /// Installs the host-owned lossless transcript conversion.
    pub fn codec(mut self, codec: Arc<dyn TranscriptCodec<C>>) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Installs optional host preparation/observation hooks.
    pub fn hooks(mut self, hooks: Arc<dyn SessionHooks<C>>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Freezes the prefix used to initialize this session's history.
    pub fn prefix(mut self, prefix: PrefixSnapshot) -> Self {
        self.prefix = prefix;
        self
    }

    /// Freezes the tool declarations exposed to each driver invocation.
    pub fn tool_snapshot(mut self, tools: ToolSnapshot) -> Self {
        self.tools = tools;
        self
    }

    /// Enables append-only transcript persistence through a session-owned
    /// locator, stem, and neutral metadata seed.
    pub fn transcript(
        mut self,
        locator: Arc<dyn TranscriptLocator>,
        stem: impl Into<String>,
        meta: TranscriptMeta,
    ) -> Self {
        self.transcript = Some(TranscriptConfig {
            locator,
            stem: stem.into(),
            meta,
        });
        self
    }

    /// Builds a session. A codec is required only when transcript persistence
    /// or transcript resume is configured.
    pub fn build(self) -> Result<Session<C>, RuntimeError> {
        let target = self.transcript.map(|config| crate::TranscriptTarget {
            locator: config.locator,
            stem: config.stem,
            resume_agent: None,
            meta: config.meta,
        });
        if target.is_some() && self.codec.is_none() {
            return Err(RuntimeError::MissingDependency("TranscriptCodec"));
        }
        Ok(Session::<C>::new(
            self.driver,
            self.codec,
            self.hooks,
            self.prefix,
            self.tools,
            target,
        ))
    }
}
