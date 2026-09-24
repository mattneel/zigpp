// The sidebar's table of contents is filled in by mdbook's own script, and it
// only knows about chapters of the book. The site also serves two things that
// are not chapters: the standard library documentation at /std/ and the
// language reference at /langref.html. This adds links to both under the table
// of contents, which keeps the theme's index.hbs untouched. The files next to
// this one in theme/ are copied to the root of the site by the site workflow.
(function () {
    const links = [
        ["Book", "/index.html"],
        ["Standard library", "/std/index.html"],
        ["Language reference", "/langref.html"],
    ];

    function addNavigation() {
        const sidebar = document.getElementById("mdbook-sidebar");
        const scrollbox = sidebar && sidebar.querySelector("mdbook-sidebar-scrollbox");
        if (!scrollbox || sidebar.querySelector(".zigpp-navigation")) return;

        const navigation = document.createElement("nav");
        navigation.className = "zigpp-navigation";
        navigation.setAttribute("aria-label", "Zig++ documentation");

        const heading = document.createElement("span");
        heading.className = "zigpp-navigation-heading";
        heading.textContent = "Documentation";
        navigation.appendChild(heading);

        for (const [label, href] of links) {
            const link = document.createElement("a");
            link.href = href;
            link.textContent = label;
            navigation.appendChild(link);
        }

        // Inside the scrollbox, so it sits under the table of contents and
        // scrolls with it.
        scrollbox.appendChild(navigation);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", addNavigation);
    } else {
        addNavigation();
    }
})();
