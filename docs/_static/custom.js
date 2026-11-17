document.addEventListener("DOMContentLoaded", function() {
    var links = document.querySelectorAll('a');
    links.forEach(function(link) {
        if (link.href.toLowerCase().endsWith('.pdf')) {
            link.removeAttribute('download');
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });
});

// Sidebar: "[en, ja]" after the entries that have a Japanese version. The Japanese pages are not in the table of
// contents (orphan); while one is open, its entry is marked as the current page. To add a pair, add a line to
// LANGUAGE_PAIRS (paths from the root of the docs).
var LANGUAGE_PAIRS = {
    "src/cw.html": "src/cw_ja.html",
    "src/multipie.html": "src/multipie_ja.html",
    "src/scw.html": "src/scw_ja.html",
    "src/format/cwin.html": "src/format/cwin_ja.html",
    "src/format/cwout.html": "src/format/cwout_ja.html",
    "src/format/hdf5.html": "src/format/hdf5_ja.html",
    "src/format/hr.html": "src/format/hr_ja.html",
    "src/format/sr.html": "src/format/sr_ja.html",
    "src/format/z.html": "src/format/z_ja.html",
    "src/format/s.html": "src/format/s_ja.html",
    "src/format/n.html": "src/format/n_ja.html",
    "src/format/convention.html": "src/format/convention_ja.html",
    "src/tutorials/tutorial01.html": "src/tutorials/tutorial01_ja.html",
    "src/tutorials/tutorial02.html": "src/tutorials/tutorial02_ja.html",
};

document.addEventListener("DOMContentLoaded", function() {
    var root = new URL(document.documentElement.dataset.content_root || "./", location.href);
    var here = location.pathname;
    document.querySelectorAll('.bd-sidebar-primary nav.bd-links a.reference.internal').forEach(function(link) {
        var target = new URL(link.getAttribute('href'), location.href).pathname;
        Object.keys(LANGUAGE_PAIRS).forEach(function(en) {
            var enUrl = new URL(en, root);
            if (target !== enUrl.pathname) {
                return;
            }
            var jaUrl = new URL(LANGUAGE_PAIRS[en], root);
            var span = document.createElement('span');
            span.className = 'language-links';
            [["en", enUrl], ["ja", jaUrl]].forEach(function(item, i) {
                if (i > 0) {
                    span.appendChild(document.createTextNode(", "));
                }
                var a = document.createElement('a');
                a.href = item[1].href;
                a.textContent = item[0];
                a.lang = item[0];
                if (here === item[1].pathname) {
                    a.className = 'current-language';
                }
                span.appendChild(a);
            });
            span.insertBefore(document.createTextNode("["), span.firstChild);
            span.appendChild(document.createTextNode("]"));
            link.appendChild(document.createTextNode(" "));
            link.after(span);
            if (here === jaUrl.pathname) {
                link.classList.add('current');
                link.parentElement.classList.add('current', 'active');
            }
        });
    });
});
