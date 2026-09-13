use strict;
use warnings;

my $file = 'cja_submission_english/newswarm_cja_english.tex';
open my $in, '<', $file or die "Cannot read $file: $!";
my @lines = <$in>;
close $in;

my $in_bibliography = 0;
for my $line (@lines) {
    $in_bibliography = 1 if $line =~ /\\begin\{thebibliography\}/;
    next if $in_bibliography;
    next if $line =~ /^\s*\\(?:label|includegraphics)/;

    $line =~ s/\bAircraft swarms\b/UAV swarms/g;
    $line =~ s/\bAircraft Swarms\b/UAV Swarms/g;
    $line =~ s/\baircraft swarms\b/UAV swarms/g;
    $line =~ s/\baircraft swarm\b/UAV swarm/g;
    $line =~ s/\bAircraft\b/UAV/g;
    $line =~ s/\baircraft\b/UAV/g;
    $line =~ s/\bmulti-UAV\b/multiple UAV/g;

    $line =~ s/\bmain-attack\b/primary-attack/g;
    $line =~ s/\bMain-attack\b/Primary-attack/g;
    $line =~ s/\bmain attack\b/primary attack/g;
    $line =~ s/\bMain attack\b/Primary attack/g;

    $line =~ s/\bmaximum normal-overload capability\b/maximum normal-acceleration capability/g;
    $line =~ s/\bnormal-overload capability\b/normal-acceleration capability/g;
    $line =~ s/\boverload-command saturation\b/acceleration-command saturation/g;
    $line =~ s/\bmaximum-overload use\b/maximum-acceleration use/g;
    $line =~ s/\bhigh-overload\b/high-acceleration/g;
    $line =~ s/\boverload capability\b/acceleration capability/g;
    $line =~ s/\bmaneuver overload\b/maneuver acceleration/g;
    $line =~ s/\bpitch overloads\b/pitch-plane accelerations/g;
    $line =~ s/\byaw overloads\b/lateral accelerations/g;
    $line =~ s/\bpitch overload\b/pitch-plane acceleration/g;
    $line =~ s/\byaw overload\b/lateral acceleration/g;
    $line =~ s/\boverloads\b/accelerations/g;
    $line =~ s/\boverload\b/acceleration/g;

    $line =~ s/\bcapture regions\b/acquisition regions/g;
    $line =~ s/\bcapture-risk identification\b/acquisition-risk identification/g;
    $line =~ s/\bcapture risk\b/acquisition risk/g;
    $line =~ s/\bcapture-penalty function\b/acquisition-risk penalty function/g;
    $line =~ s/\bcapture penalty\b/acquisition-risk penalty/g;
    $line =~ s/\bcapture intensity\b/engagement intensity/g;
    $line =~ s/\bcapture probability\b/engagement likelihood/g;
    $line =~ s/\bcapture-probability\b/engagement-likelihood/g;
    $line =~ s/\bactual number of captures\b/number of active interceptor assignments/g;
    $line =~ s/\bcaptured target\b/assigned target/g;
    $line =~ s/\bcurrently captured by enemy interceptors\b/currently engaged by enemy interceptors/g;
    $line =~ s/\bcurrently capturing\b/currently engaging/g;
    $line =~ s/\bactively attracting capture\b/actively drawing interceptor engagements/g;
    $line =~ s/\bactively attracts capture\b/actively draws interceptor engagements/g;
    $line =~ s/\battracts capture\b/draws interceptor engagements/g;
    $line =~ s/\bcaptured by the defensive swarm\b/engaged by the interceptor swarm/g;
    $line =~ s/\bcontinuous capture\b/continuous engagement/g;

    $line =~ s/\bgame confrontation\b/adversarial engagement/g;
    $line =~ s/\bdynamic-game confrontation\b/dynamic adversarial engagement/g;
    $line =~ s/\boffensive--defensive confrontation\b/offensive--defensive engagement/g;
    $line =~ s/\btask dynamic-reconfiguration\b/dynamic task reconfiguration/g;
    $line =~ s/\bTask Dynamic Reconfiguration\b/Dynamic Task Reconfiguration/g;
    $line =~ s/\btask dynamic reconfiguration\b/dynamic task reconfiguration/g;
    $line =~ s/Maneuvering penetration mainly reduces the probability of detection, capture, and successful interception by actively changing the flight trajectory and line-of-sight geometry\./Maneuvering penetration reduces the probabilities of detection, target acquisition, and successful interception by actively modifying the flight trajectory and line-of-sight geometry./g;
    $line =~ s/\blow-capture-risk\b/low-acquisition-risk/g;
    $line =~ s/\bteammate capture count\b/normalized number of engaged teammates/g;
    $line =~ s/\bmultiple UAV under\b/multiple UAVs under/g;
    $line =~ s/\boffensive UAV with\b/offensive UAVs with/g;
    $line =~ s/\bmultiple UAV\b/multiple UAVs/g;
    $line =~ s/\bhit-decision threshold\b/kill-radius threshold/g;
    $line =~ s/\bpenetration-success degree\b/penetration-success metric/g;
    $line =~ s/\bstrike-feasibility degree\b/strike-feasibility metric/g;
    $line =~ s/To deeply incorporate the swarm-motion prior/To incorporate the swarm-motion prior/g;
    $line =~ s/have lower rising speeds and lower final reward levels/exhibit slower reward growth and lower final rewards/g;
    $line =~ s/remains at a relatively low and weakly fluctuating error level/remains low with limited fluctuations/g;
    $line =~ s/This phenomenon indicates that the analytical priors/This behavior indicates that the analytical priors/g;
}

open my $out, '>', $file or die "Cannot write $file: $!";
print {$out} @lines;
close $out;
