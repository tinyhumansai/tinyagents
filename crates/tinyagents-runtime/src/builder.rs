use std::sync::Arc;
use tinyagents_harness::tool::ToolSnapshot;

use tinyagents_session::transcript::{SessionRef, TranscriptLocator, TranscriptMeta, session_stem};

use crate::SessionIdentityResolver;
use crate::identity::IdentityConfig;
use crate::{
    NoopSessionHooks, PrefixSnapshot, RuntimeError, Session, SessionDriver, SessionHooks,
    TranscriptCodec,
};

/// Configures a directly-owned, reusable [`Session`].
pub struct SessionBuilder<C: Clone + Send + Sync + 'static = ()> {
    driver: Arc<dyn SessionDriver<C>>,
    codec: Option<Arc<dyn TranscriptCodec<C>>>,
    hooks: Arc<dyn SessionHooks<C>>,
    post_commit_sinks: Vec<Arc<dyn crate::hooks::PostCommitSink<C>>>,
    identity: Option<IdentityConfig<C>>,
    prefix: PrefixSnapshot,
    tools: ToolSnapshot,
    retain_recorded_tools: bool,
    prune_retired_tools: bool,
    transcript: Option<TranscriptConfig>,
}

struct TranscriptConfig {
    locator: Arc<dyn TranscriptLocator>,
    stem: String,
    session: Option<SessionRef>,
    resume_agent: Option<String>,
    meta: TranscriptMeta,
}

impl<C: Clone + Send + Sync + 'static> SessionBuilder<C> {
    /// Starts a builder over an object-safe execution driver.
    pub fn new(driver: Arc<dyn SessionDriver<C>>) -> Self {
        Self {
            driver,
            codec: None,
            hooks: Arc::new(NoopSessionHooks),
            post_commit_sinks: Vec::new(),
            identity: None,
            prefix: PrefixSnapshot::default(),
            tools: ToolSnapshot::default(),
            retain_recorded_tools: false,
            prune_retired_tools: false,
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

    /// Adds a host observer for successful durable commits.
    pub fn post_commit_sink(mut self, sink: Arc<dyn crate::hooks::PostCommitSink<C>>) -> Self {
        self.post_commit_sinks.push(sink);
        self
    }

    /// Resolves a root session's logical key against host-authorized storage.
    pub fn identity_resolver(
        mut self,
        session_key: impl Into<String>,
        meta: TranscriptMeta,
        resolver: Arc<dyn SessionIdentityResolver<C>>,
    ) -> Self {
        self.identity = Some(IdentityConfig {
            resolver,
            session_key: session_key.into(),
            parent: None,
            child_key: None,
            meta,
        });
        self
    }

    /// Resolves a child session while preserving its logical parent lineage.
    pub fn child_identity_resolver(
        mut self,
        parent: SessionRef,
        child_key: impl Into<String>,
        meta: TranscriptMeta,
        resolver: Arc<dyn SessionIdentityResolver<C>>,
    ) -> Self {
        let child_key = child_key.into();
        self.identity = Some(IdentityConfig {
            resolver,
            session_key: child_key.clone(),
            parent: Some(parent),
            child_key: Some(child_key),
            meta,
        });
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

    /// Keeps every tool declaration this session has already sent, even when a
    /// later turn's preparation no longer supplies it.
    ///
    /// A session always records the declarations each turn was sent with
    /// (`{"kind":"tools"}` in its transcript) and restores them on resume.
    /// With retention on, a declaration the host stops supplying — typically
    /// because a new process has not rebuilt it yet — is merged back in rather
    /// than disappearing from the model's tool list mid-conversation, which
    /// would both break the prompt's own references to it and change the
    /// cached request prefix. The host must then be able to execute (or
    /// refuse) a retained declaration; it can read them from
    /// [`Session::recorded_tools`](crate::Session::recorded_tools) after
    /// resuming a session.
    ///
    /// Off by default because a driver that requires the snapshot to equal
    /// its own registry (`HarnessDriver`) cannot execute a retained entry.
    pub fn retain_recorded_tools(mut self, retain: bool) -> Self {
        self.retain_recorded_tools = retain;
        self
    }

    /// With [`Self::retain_recorded_tools`] on, keeps a recorded declaration
    /// only while the host still offers its name, and drops the rest from the
    /// turn's tools and from the next `{"kind":"tools"}` record.
    ///
    /// Without pruning, a retired tool is merged back on every turn and
    /// recorded again, so it ratchets forward for the life of the thread and
    /// the model keeps calling a tool the host now refuses. The host's live
    /// catalogue is the turn's [`ToolSnapshot`]: its declarations plus any
    /// name added with [`ToolSnapshot::with_offered_names`], which is how a
    /// deferred (discoverable) tool stays retained.
    ///
    /// Off by default, so an existing thread keeps its recorded tools until a
    /// host opts in. It has no effect while retention is off.
    pub fn prune_retired_tools(mut self, prune: bool) -> Self {
        self.prune_retired_tools = prune;
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
            session: None,
            resume_agent: None,
            meta,
        });
        self
    }

    /// Enables persistence addressed by durable session identity.
    ///
    /// Prefer this over [`Self::transcript`]: the stem it derives is stable
    /// across processes and launches, so one conversation stays in one
    /// transcript instead of accumulating a file per cold boot. It is also what
    /// [`ResumeMode::Session`](crate::ResumeMode::Session) resolves against.
    pub fn session(
        mut self,
        locator: Arc<dyn TranscriptLocator>,
        session: SessionRef,
        mut meta: TranscriptMeta,
    ) -> Self {
        meta.session_id = Some(session.session_id());
        meta.parent_session_id = session.parent_session_id();
        self.transcript = Some(TranscriptConfig {
            locator,
            stem: session_stem(&session),
            session: Some(session),
            resume_agent: None,
            meta,
        });
        self
    }

    /// Uses a distinct agent key for `ResumeMode::LatestForAgent` lookup.
    ///
    /// Previously reachable only through a hook-supplied `ResumePreparation`,
    /// which meant a host that simply wanted a different resume key had to
    /// implement a hook to say so.
    pub fn resume_agent(mut self, resume_agent: impl Into<String>) -> Self {
        if let Some(config) = self.transcript.as_mut() {
            config.resume_agent = Some(resume_agent.into());
        }
        self
    }

    /// Builds a session. A codec is required only when transcript persistence
    /// or transcript resume is configured.
    pub fn build(self) -> Result<Session<C>, RuntimeError> {
        let target = self.transcript.map(|config| crate::TranscriptTarget {
            locator: config.locator,
            stem: config.stem,
            resume_agent: config.resume_agent,
            session: config.session,
            meta: config.meta,
        });
        let identity = self.identity;
        if (target.is_some() || identity.is_some()) && self.codec.is_none() {
            return Err(RuntimeError::MissingDependency("TranscriptCodec"));
        }
        let mut session = Session::<C>::new(
            self.driver,
            self.codec,
            self.hooks,
            self.post_commit_sinks,
            self.prefix,
            self.tools,
            target,
            identity,
        );
        session.set_retain_recorded_tools(self.retain_recorded_tools);
        session.set_prune_retired_tools(self.prune_retired_tools);
        Ok(session)
    }
}
