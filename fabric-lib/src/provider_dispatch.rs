use std::borrow::Cow;

#[cfg(feature = "efa")]
use crate::efa::EfaDomainInfo;
use crate::{provider::RdmaDomainInfo, verbs::VerbsDeviceInfo};

#[derive(Clone)]
pub enum DomainInfo {
    #[cfg(feature = "efa")]
    Efa(EfaDomainInfo),
    Verbs(VerbsDeviceInfo),
}

impl RdmaDomainInfo for DomainInfo {
    fn name(&self) -> Cow<'_, str> {
        match self {
            #[cfg(feature = "efa")]
            DomainInfo::Efa(info) => info.name(),
            DomainInfo::Verbs(info) => info.name(),
        }
    }

    fn link_speed(&self) -> u64 {
        match self {
            #[cfg(feature = "efa")]
            DomainInfo::Efa(info) => info.link_speed(),
            DomainInfo::Verbs(info) => info.link_speed(),
        }
    }
}
